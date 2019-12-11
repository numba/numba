from __future__ import print_function, absolute_import
import copy
from collections import namedtuple

from numba.typing.templates import ConcreteTemplate
from numba import types, compiler, ir
from .oneapidriver import driver
from numba.typing.templates import AbstractTemplate
from numba import ctypes_support as ctypes
from numba.oneapi.oneapidriver import spirv_generator
from types import FunctionType
import os

DEBUG=os.environ.get('NUMBA_ONEAPI_DEBUG', None)

def _raise_no_device_found_error():
    error_message = ("No OpenCL device specified. "
                     "Usage : jit_fn[device, globalsize, localsize](...)")
    raise ValueError(error_message)


def compile_with_oneapi(pyfunc, return_type, args, debug):
    # First compilation will trigger the initialization of the OpenCL backend.
    from .descriptor import OneAPITargetDesc

    typingctx = OneAPITargetDesc.typingctx
    targetctx = OneAPITargetDesc.targetctx
    # TODO handle debug flag
    flags = compiler.Flags()
    # Do not compile (generate native code), just lower (to LLVM)
    flags.set('no_compile')
    flags.set('no_cpython_wrapper')
    flags.unset('nrt')
    # Run compilation pipeline
    if isinstance(pyfunc, FunctionType):
        cres = compiler.compile_extra(typingctx=typingctx,
                                      targetctx=targetctx,
                                      func=pyfunc,
                                      args=args,
                                      return_type=return_type,
                                      flags=flags,
                                      locals={})
    elif isinstance(pyfunc, ir.FunctionIR):
        cres = compiler.compile_ir(typingctx=typingctx,
                                   targetctx=targetctx,
                                   func_ir=pyfunc,
                                   args=args,
                                   return_type=return_type,
                                   flags=flags,
                                   locals={})
    else:
        assert(0)

    # Linking depending libraries
    # targetctx.link_dependencies(cres.llvm_module, cres.target_context.linking)
    library = cres.library
    library.finalize()

    return cres


def compile_kernel(device, pyfunc, args, debug=False):
    if DEBUG:
        print("compile_kernel", args)
    cres = compile_with_oneapi(pyfunc, types.void, args, debug=debug)
    func = cres.library.get_function(cres.fndesc.llvm_func_name)
    kernel = cres.target_context.prepare_ocl_kernel(func, cres.signature.args)
    oclkern = OneAPIKernel(device_env=device,
                           llvm_module=kernel.module,
                           name=kernel.name,
                           argtypes=cres.signature.args)
    return oclkern


def compile_kernel_parfor(device, func_ir, args, debug=False):
    if DEBUG:
        print("compile_kernel_parfor", args)
    cres = compile_with_oneapi(func_ir, types.void, args, debug=debug)
    func = cres.library.get_function(cres.fndesc.llvm_func_name)
    kernel = cres.target_context.prepare_ocl_kernel(func, cres.signature.args)
    oclkern = OneAPIKernel(device_env=device,
                           llvm_module=kernel.module,
                           name=kernel.name,
                           argtypes=cres.signature.args)
    return oclkern

def _ensure_list(val):
    if not isinstance(val, (tuple, list)):
        return [val]
    else:
        return list(val)


def _ensure_size_or_append(val, size):
    n = len(val)
    for _ in range(n, size):
        val.append(1)


class OneAPIKernelBase(object):
    """Define interface for configurable kernels
    """

    def __init__(self):
        self.global_size = (1,)
        self.local_size = (1,)
        self.device_env = None

    def copy(self):
        return copy.copy(self)

    def configure(self, device_env, global_size, local_size=None):
        """Configure the OpenCL kernel local_size can be None
        """
        global_size = _ensure_list(global_size)

        if local_size is not None:
            local_size = _ensure_list(local_size)
            size = max(len(global_size), len(local_size))
            _ensure_size_or_append(global_size, size)
            _ensure_size_or_append(local_size, size)

        clone = self.copy()
        clone.global_size = tuple(global_size)
        clone.local_size = tuple(local_size) if local_size else None
        clone.device_env = device_env

        return clone

    def forall(self, nelem, local_size=64, queue=None):
        """Simplified configuration for 1D kernel launch
        """
        return self.configure(nelem, min(nelem, local_size), queue=queue)

    def __getitem__(self, args):
        """Mimick CUDA python's square-bracket notation for configuration.
        This assumes the argument to be:
            `griddim, blockdim, queue`
        The blockdim maps directly to local_size.
        The actual global_size is computed by multiplying the local_size to
        griddim.
        """

        device_env = args[0]
        griddim = _ensure_list(args[1])
        blockdim = _ensure_list(args[2])
        size = max(len(griddim), len(blockdim))
        _ensure_size_or_append(griddim, size)
        _ensure_size_or_append(blockdim, size)
        # Compute global_size
        gs = [g * l for g, l in zip(griddim, blockdim)]
        return self.configure(device_env, gs, blockdim)


#_CacheEntry = namedtuple("_CachedEntry", ['symbol', 'executable',
#                                          'kernarg_region'])


# class _CachedProgram(object):
#    def __init__(self, entry_name, binary):
#        self._entry_name = entry_name
#        self._binary = binary
#        # key: ocl context
#        self._cache = {}
#
#    def get(self, device):
#        context = device.get_context()
#        result = self._cache.get(context)
#
#        if result is not None:
#            program = result[1]
#            kernel = result[2]
#        else:
#            # First-time compilation
#            spirv_bc = spirv.llvm_to_spirv(self._binary)
#            program = context.create_program_from_il(spirv_bc)
#            program.build()
#            kernel = program.create_kernel(self._entry_name)
#
#            # Cache the just built cl_program, its cl_device and a cl_kernel
#            self._cache[context] = (device, program, kernel)
#
#        return context, device, program, kernel


class OneAPIKernel(OneAPIKernelBase):
    """
    A OCL kernel object
    """

    def __init__(self, device_env, llvm_module, name, argtypes):
        super(OneAPIKernel, self).__init__()
        self._llvm_module = llvm_module
        self.assembly = self.binary = llvm_module.__str__()
        self.entry_name = name
        self.argument_types = tuple(argtypes)
        self._argloc = []
        # cached finalized program
        # self._cacheprog = _CachedProgram(entry_name=self.entry_name,
        #                                 binary=self.binary)
        # First-time compilation using SPIRV-Tools
        self.spirv_bc = spirv_generator.llvm_to_spirv(self.binary)
        #print("OneAPIKernel:", self.spirv_bc, type(self.spirv_bc))
        # create a program
        self.program = driver.Program(device_env, self.spirv_bc)
        #  create a kernel
        self.kernel = driver.Kernel(device_env, self.program, self.entry_name)
    # def bind(self):
    #    """
    #    Bind kernel to device
    #    """
    #    return self._cacheprog.get(self.device)

    def __call__(self, *args):

        # Create an array of KenrelArgs
        # Unpack pyobject values into ctypes scalar values
        retr = []  # hold functors for writeback
        kernelargs = []
        for ty, val in zip(self.argument_types, args):
            self._unpack_argument(ty, val, self.device_env, retr, kernelargs)
        
        # enqueues the kernel
        driver.enqueue_kernel(self.device_env, self.kernel, kernelargs,
                              self.global_size, self.local_size)

        # retrieve auto converted arrays
        # for wb in retr:
        #    wb()
    
    def _unpack_device_array_argument(self, val, kernelargs):
        void_ptr_arg = True
        # meminfo 
        kernelargs.append(driver.KernelArg(None, void_ptr_arg))
        # parent
        kernelargs.append(driver.KernelArg(None, void_ptr_arg))
        kernelargs.append(driver.KernelArg(val._ndarray.size))
        kernelargs.append(driver.KernelArg(val._ndarray.dtype.itemsize))
        kernelargs.append(driver.KernelArg(val))
        for ax in range(val._ndarray.ndim):
            kernelargs.append(driver.KernelArg(val._ndarray.shape[ax]))
        for ax in range(val._ndarray.ndim):
            kernelargs.append(driver.KernelArg(val._ndarray.strides[ax]))


    def _unpack_argument(self, ty, val, queue, retr, kernelargs):
        """
        Convert arguments to ctypes and append to kernelargs
        """
        # DRD : Check if the val is of type driver.DeviceArray before checking
        # if ty is of type ndarray. Argtypes retuends ndarray for both
        # DeviceArray and ndarray. This is a hack to get around the issue,
        # till I understand the typing infrastructure of NUMBA better.
        if isinstance(val, driver.DeviceArray):
            _unpack_device_array_argument(val, kernelargs)

        elif isinstance(ty, types.Array):
            # DRD: This unpack routine is commented out for the time-being.
            # NUMBA does not know anything about SmartArrayTypes.

            # if isinstance(ty, types.SmartArrayType):
            #    devary = val.get('gpu')
            #    retr.append(lambda: val.mark_changed('gpu'))
            #    outer_parent = ctypes.c_void_p(0)
            #    kernelargs.append(outer_parent)
            # else:
            #devary, conv = devicearray.auto_device(val, stream=queue)
            #if conv:
            #    retr.append(lambda: devary.copy_to_host(val, stream=queue))
            """
            c_intp = ctypes.c_ssize_t

            # TA: New version
            meminfo = ctypes.c_void_p(0)
            parent = ctypes.c_void_p(0)
            nitems = c_intp(devary.size)
            itemsize = c_intp(devary.dtype.itemsize)
            data = driver.device_pointer(devary)  # @@

            kernelargs.append(driver.runtime.create_kernel_arg(meminfo))
            kernelargs.append(driver.runtime.create_kernel_arg(parent))
            kernelargs.append(driver.runtime.create_kernel_arg(nitems))
            kernelargs.append(driver.runtime.create_kernel_arg(itemsize))
            kernelargs.append(driver.runtime.create_kernel_arg(data))

            meminfo = ctypes.c_void_p(0)
            parent = ctypes.c_void_p(0)
            nitems = c_intp(devary.size)
            itemsize = c_intp(devary.dtype.itemsize)
            data = driver.device_pointer(devary)  # @@
            kernelargs.append(meminfo)
            kernelargs.append(parent)
            kernelargs.append(nitems)
            kernelargs.append(itemsize)
            kernelargs.append(data)
            for ax in range(devary.ndim):
                kernelargs.append(c_intp(devary.shape[ax]))
            for ax in range(devary.ndim):
                kernelargs.append(c_intp(devary.strides[ax]))
            """
            raise NotImplementedError(ty, val)

        elif isinstance(ty, types.Integer):
            cval = getattr(ctypes, "c_%s" % ty)(val)
            #kernelargs.append(cval)
            raise NotImplementedError(ty, val)

        elif ty == types.float64:
            cval = ctypes.c_double(val)
            #kernelargs.append(cval)
            raise NotImplementedError(ty, val)

        elif ty == types.float32:
            cval = ctypes.c_float(val)
            #kernelargs.append(cval)
            raise NotImplementedError(ty, val)

        elif ty == types.boolean:
            cval = ctypes.c_uint8(int(val))
            #kernelargs.append(cval)
            raise NotImplementedError(ty, val)

        elif ty == types.complex64:
            #kernelargs.append(ctypes.c_float(val.real))
            #kernelargs.append(ctypes.c_float(val.imag))
            raise NotImplementedError(ty, val)


        elif ty == types.complex128:
            #kernelargs.append(ctypes.c_double(val.real))
            #kernelargs.append(ctypes.c_double(val.imag))
            raise NotImplementedError(ty, val)


        else:
            raise NotImplementedError(ty, val)


class AutoJitOneAPIKernel(OneAPIKernelBase):
    def __init__(self, func):

        super(AutoJitOneAPIKernel, self).__init__()
        self.py_func = func
        self.definitions = {}

        from .descriptor import OneAPITargetDesc

        self.typingctx = OneAPITargetDesc.typingctx

    def __call__(self, *args):
        if self.device_env is None:
            _raise_no_device_found_error()
        kernel = self.specialize(*args)
        cfg = kernel.configure(self.device_env, self.global_size, 
                               self.local_size)
        cfg(*args)

    def specialize(self, *args):
        argtypes = tuple([self.typingctx.resolve_argument_type(a)
                          for a in args])
        kernel = self.definitions.get(argtypes)

        if kernel is None:
            kernel = compile_kernel(self.device_env, self.py_func, argtypes)
            self.definitions[argtypes] = kernel
        return kernel
