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


def compile_kernel(device, pyfunc, args, access_types, debug=False):
    if DEBUG:
        print("compile_kernel", args)
    cres = compile_with_dppy(pyfunc, types.void, args, debug=debug)
    func = cres.library.get_function(cres.fndesc.llvm_func_name)
    kernel = cres.target_context.prepare_ocl_kernel(func, cres.signature.args)
    oclkern = DPPyKernel(device_env=device,
                         llvm_module=kernel.module,
                         name=kernel.name,
                         argtypes=cres.signature.args,
                         ordered_arg_access_types=access_types)
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

def _ensure_list(val):
    if not isinstance(val, (tuple, list)):
        return [val]
    else:
        return list(val)


def _ensure_size_or_append(val, size):
    n = len(val)
    for _ in range(n, size):
        val.append(1)


class DPPyKernelBase(object):
    """Define interface for configurable kernels
    """

    def __init__(self):
        self.global_size = (1,)
        self.local_size = (1,)
        self.device_env = None

        # list of supported access types, stored in dict for fast lookup
        self.correct_access_types = {"read_only": 0, "write_only": 1, "read_write": 2}

    def copy(self):
        return copy.copy(self)

    def configure(self, device_env, global_size, local_size=None):
        """Configure the OpenCL kernel. The local_size can be None
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
            `dppy_driver.device_env, global size, local size`
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


class DPPyKernel(DPPyKernelBase):
    """
    A OCL kernel object
    """

    def __init__(self, device_env, llvm_module, name, argtypes, ordered_arg_access_types):
        super(DPPyKernel, self).__init__()
        self._llvm_module = llvm_module
        self.assembly = self.binary = llvm_module.__str__()
        self.entry_name = name
        self.argument_types = tuple(argtypes)
        self.ordered_arg_access_types = ordered_arg_access_types
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
        internal_device_arrs = []
        for ty, val, access_type in zip(self.argument_types, args, self.ordered_arg_access_types):
            self._unpack_argument(ty, val, self.device_env, retr,
                    kernelargs, internal_device_arrs, access_type)

        # enqueues the kernel
        driver.enqueue_kernel(self.device_env, self.kernel, kernelargs,
                              self.global_size, self.local_size)

        for ty, val, i_dev_arr, access_type in zip(self.argument_types, args,
                internal_device_arrs, self.ordered_arg_access_types):
            self._pack_argument(ty, val, self.device_env, i_dev_arr, access_type)
        # retrieve auto converted arrays
        # for wb in retr:
        #    wb()

    def _pack_argument(self, ty, val, device_env, device_arr, access_type):
        """
        Copy device data back to host
        """
        if device_arr and (access_type not in self.correct_access_types or \
            access_type in self.correct_access_types and \
            self.correct_access_types[access_type] != 0):
            # we get the date back to host if have created a device_array or
            # if access_type of this device_array is not of type read_only and read_write
            device_env.copy_array_from_device(device_arr)

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


    def _unpack_argument(self, ty, val, device_env, retr, kernelargs, device_arrs, access_type):
        """
        Convert arguments to ctypes and append to kernelargs
        """
        # DRD : Check if the val is of type driver.DeviceArray before checking
        # if ty is of type ndarray. Argtypes returns ndarray for both
        # DeviceArray and ndarray. This is a hack to get around the issue,
        # till I understand the typing infrastructure of NUMBA better.
        if isinstance(val, driver.DeviceArray):
            self._unpack_device_array_argument(val, kernelargs)
            device_arrs.append(None)

        elif isinstance(ty, types.Array):
            default_behavior = self.check_and_warn_abt_invalid_access_type(access_type)

            dArr = None
            if default_behavior or \
            self.correct_access_types[access_type] == 0 or \
            self.correct_access_types[access_type] == 2:
                # default, read_only and read_write case
                dArr = device_env.copy_array_to_device(val)
            elif self.correct_access_types[access_type] == 1:
                # write_only case, we do not copy the host data
                print("-------Only creating buff not copying from host----")
                dArr = driver.DeviceArray(device_env.get_env_ptr(), val)

            assert (dArr != None), "Problem in allocating device buffer"
            device_arrs.append(dArr)
            self._unpack_device_array_argument(dArr, kernelargs)

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

    def check_and_warn_abt_invalid_access_type(self, access_type):
        if access_type not in self.correct_access_types:
            msg = "[!] %s is not a valid access type. Supported access types are [" % (access_type)
            for key in self.correct_access_types:
                msg += " %s |" % (key)

            msg = msg[:-1] + "]"
            if access_type != None: print(msg)
            return True
        else:
            return False


class JitDPPyKernel(DPPyKernelBase):
    def __init__(self, func, access_types):

        super(AutoJitDPPyKernel, self).__init__()

        self.py_func = func
        # DRD: Caching definitions this way can lead to unexpected consequences
        # E.g. A kernel compiled for a given device would not get recompiled
        # and lead to OpenCL runtime errors.
        #self.definitions = {}
        self.access_types = access_types

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
            kernel = compile_kernel(self.device_env, self.py_func, argtypes, self.access_types)
            #self.definitions[argtypes] = kernel
        return kernel
