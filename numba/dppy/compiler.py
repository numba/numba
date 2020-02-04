from __future__ import print_function, absolute_import
import copy
from collections import namedtuple

from numba.typing.templates import ConcreteTemplate
from numba import types, compiler, ir
from .dppy_driver import driver
from numba.typing.templates import AbstractTemplate
from numba import ctypes_support as ctypes
from numba.dppy.dppy_driver import spirv_generator
from types import FunctionType
import os

DEBUG=os.environ.get('NUMBA_DPPY_DEBUG', None)

def _raise_no_device_found_error():
    error_message = ("No OpenCL device specified. "
                     "Usage : jit_fn[device, globalsize, localsize](...)")
    raise ValueError(error_message)

def _raise_invalid_kernel_enqueue_args():
    error_message = ("Incorrect number of arguments for enquing dppy.kernel. "
                     "Usage: dppy_driver.device_env, global size, local size. "
                     "The local size argument is optional.")
    raise ValueError(error_message)

def compile_with_dppy(pyfunc, return_type, args, debug):
    # First compilation will trigger the initialization of the OpenCL backend.
    from .descriptor import DPPyTargetDesc

    typingctx = DPPyTargetDesc.typingctx
    targetctx = DPPyTargetDesc.targetctx
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
    cres = compile_with_dppy(pyfunc, types.void, args, debug=debug)
    func = cres.library.get_function(cres.fndesc.llvm_func_name)
    kernel = cres.target_context.prepare_ocl_kernel(func, cres.signature.args)
    oclkern = DPPyKernel(device_env=device,
                         llvm_module=kernel.module,
                         name=kernel.name,
                         argtypes=cres.signature.args)
    return oclkern


def compile_kernel_parfor(device, func_ir, args, debug=False):
    if DEBUG:
        print("compile_kernel_parfor", args)
    cres = compile_with_dppy(func_ir, types.void, args, debug=debug)
    func = cres.library.get_function(cres.fndesc.llvm_func_name)
    kernel = cres.target_context.prepare_ocl_kernel(func, cres.signature.args)
    oclkern = DPPyKernel(device_env=device,
                         llvm_module=kernel.module,
                         name=kernel.name,
                         argtypes=cres.signature.args)
    return oclkern


def _ensure_valid_work_item_grid(val, device_env):

    if not isinstance(val, (tuple, list)):
        error_message = ("Cannot create work item dimension from "
                         "provided argument")
        raise ValueError(error_message)

    if len(val) > device_env.get_max_work_item_dims():
        error_message = ("Unsupported number of work item dimensions ")
        raise ValueError(error_message)

    return list(val)

def _ensure_valid_work_group_size(val, work_item_grid):

    if not isinstance(val, (tuple, list)):
        error_message = ("Cannot create work item dimension from "
                         "provided argument")
        raise ValueError(error_message)

    if len(val) != len(work_item_grid):
        error_message = ("Unsupported number of work item dimensions ")
        raise ValueError(error_message)

    return list(val)


class DPPyKernelBase(object):
    """Define interface for configurable kernels
    """

    def __init__(self):
        self.global_size = []
        self.local_size  = []
        self.device_env  = None

    def copy(self):
        return copy.copy(self)

    def configure(self, device_env, global_size, local_size=None):
        """Configure the OpenCL kernel. The local_size can be None
        """
        clone = self.copy()
        clone.global_size = global_size
        clone.local_size = local_size if local_size else []
        clone.device_env = device_env

        return clone

    def forall(self, nelem, local_size=64, queue=None):
        """Simplified configuration for 1D kernel launch
        """
        return self.configure(nelem, min(nelem, local_size), queue=queue)

    def __getitem__(self, args):
        """Mimick CUDA python's square-bracket notation for configuration.
        This assumes the argument to be:
            `dppy_driver.device_env, global size, local size`
        """
        ls = None
        nargs = len(args)
        # Check if the kernel enquing arguments are sane
        if nargs < 2 or nargs > 3:
            _raise_invalid_kernel_enqueue_args

        device_env = args[0]
        gs = _ensure_valid_work_item_grid(args[1], device_env)
        # If the optional local size argument is provided
        if nargs == 3:
            ls = _ensure_valid_work_group_size(args[2], gs)

        return self.configure(device_env, gs, ls)


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


class DPPyKernel(DPPyKernelBase):
    """
    A OCL kernel object
    """

    def __init__(self, device_env, llvm_module, name, argtypes):
        super(DPPyKernel, self).__init__()
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
        #print("DPPyKernel:", self.spirv_bc, type(self.spirv_bc))
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
        
        # enqueue the kernel
        driver.enqueue_kernel(self.device_env, self.kernel, kernelargs,
                              self.global_size, self.local_size)


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


    def _unpack_argument(self, ty, val, device_env, retr, kernelargs):
        """
        Convert arguments to ctypes and append to kernelargs
        """
        # DRD : Check if the val is of type driver.DeviceArray before checking
        # if ty is of type ndarray. Argtypes returns ndarray for both
        # DeviceArray and ndarray. This is a hack to get around the issue,
        # till I understand the typing infrastructure of NUMBA better.
        if isinstance(val, driver.DeviceArray):
            self._unpack_device_array_argument(val, kernelargs)

        elif isinstance(ty, types.Array):
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


class JitDPPyKernel(DPPyKernelBase):

    def __init__(self, func):
        super(JitDPPyKernel, self).__init__()
        self.py_func = func
        # DRD: Caching definitions this way can lead to unexpected consequences
        # E.g. A kernel compiled for a given device would not get recompiled
        # and lead to OpenCL runtime errors.
        #self.definitions = {}

        from .descriptor import DPPyTargetDesc

        self.typingctx = DPPyTargetDesc.typingctx

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
        kernel = None #self.definitions.get(argtypes)

        if kernel is None:
            kernel = compile_kernel(self.device_env, self.py_func, argtypes)
            #self.definitions[argtypes] = kernel
        return kernel
