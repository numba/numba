from __future__ import print_function, absolute_import
import copy
from collections import namedtuple

from numba.typing.templates import ConcreteTemplate
from numba import types, compiler
from .ocldrv import devices, devicearray, driver
from numba.typing.templates import AbstractTemplate
from numba import ctypes_support as ctypes
from numba.ocl.ocldrv import spirv
from numba.ocl.ocldrv import spir2


def compile_ocl(pyfunc, return_type, args, debug):
    # First compilation will trigger the initialization of the OpenCL backend.
    from .descriptor import OCLTargetDesc

    typingctx = OCLTargetDesc.typingctx
    targetctx = OCLTargetDesc.targetctx
    # TODO handle debug flag
    flags = compiler.Flags()
    # Do not compile (generate native code), just lower (to LLVM)
    flags.set('no_compile')
    flags.set('no_cpython_wrapper')
    flags.unset('nrt')
    # Run compilation pipeline
    cres = compiler.compile_extra(typingctx=typingctx,
                                  targetctx=targetctx,
                                  func=pyfunc,
                                  args=args,
                                  return_type=return_type,
                                  flags=flags,
                                  locals={})

    # Linking depending libraries
    # targetctx.link_dependencies(cres.llvm_module, cres.target_context.linking)
    library = cres.library
    library.finalize()

    return cres


def compile_kernel(pyfunc, args, debug=False):
    cres = compile_ocl(pyfunc, types.void, args, debug=debug)
    func = cres.library.get_function(cres.fndesc.llvm_func_name)
    kernel = cres.target_context.prepare_ocl_kernel(func, cres.signature.args)
    oclkern = OCLKernel(llvm_module=kernel.module,
                        name=kernel.name,
                        argtypes=cres.signature.args)
    return oclkern


def compile_device(pyfunc, return_type, args, debug=False):
    cres = compile_ocl(pyfunc, return_type, args, debug=debug)
    func = cres.library.get_function(cres.fndesc.llvm_func_name)
    cres.target_context.mark_ocl_device(func)
    devfn = DeviceFunction(cres)

    class device_function_template(ConcreteTemplate):
        key = devfn
        cases = [cres.signature]

    cres.typing_context.insert_user_function(devfn, device_function_template)
    libs = [cres.library]
    cres.target_context.insert_user_function(devfn, cres.fndesc, libs)
    return devfn


def compile_device_template(pyfunc):
    """Compile a DeviceFunctionTemplate
    """
    from .descriptor import OCLTargetDesc

    dft = DeviceFunctionTemplate(pyfunc)

    class device_function_template(AbstractTemplate):
        key = dft

        def generic(self, args, kws):
            assert not kws
            return dft.compile(args)

    typingctx = OCLTargetDesc.typingctx
    typingctx.insert_user_function(dft, device_function_template)
    return dft


class DeviceFunctionTemplate(object):
    """Unmaterialized device function
    """
    def __init__(self, pyfunc, debug=False):
        self.py_func = pyfunc
        self.debug = debug
        # self.inline = inline
        self._compileinfos = {}

    def compile(self, args):
        """Compile the function for the given argument types.

        Each signature is compiled once by caching the compiled function inside
        this object.
        """
        if args not in self._compileinfos:
            cres = compile_ocl(self.py_func, None, args, debug=self.debug)
            func = cres.library.get_function(cres.fndesc.llvm_func_name)
            cres.target_context.mark_ocl_device(func)
            first_definition = not self._compileinfos
            self._compileinfos[args] = cres
            libs = [cres.library]

            if first_definition:
                # First definition
                cres.target_context.insert_user_function(self, cres.fndesc,
                                                         libs)
            else:
                cres.target_context.add_user_function(self, cres.fndesc, libs)

        else:
            cres = self._compileinfos[args]

        return cres.signature


class DeviceFunction(object):
    def __init__(self, cres):
        self.cres = cres


def _ensure_list(val):
    if not isinstance(val, (tuple, list)):
        return [val]
    else:
        return list(val)


def _ensure_size_or_append(val, size):
    n = len(val)
    for _ in range(n, size):
        val.append(1)


class OCLKernelBase(object):
    """Define interface for configurable kernels
    """

    def __init__(self):
        self.global_size = (1,)
        self.local_size = (1,)
        self.queue = None

    def copy(self):
        return copy.copy(self)

    def configure(self, global_size, local_size=None, queue=None):
        """Configure the OpenCL kernel
        local_size can be None
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
        clone.queue = queue

        return clone

    def forall(self, nelem, local_size=64, queue=None):
        """Simplified configuration for 1D kernel launch
        """
        return self.configure(nelem, min(nelem, local_size), queue=queue)

    def __getitem__(self, args):
        """Mimick CUDA python's square-bracket notation for configuration.
        This assumes a the argument to be:
            `griddim, blockdim, queue`
        The blockdim maps directly to local_size.
        The actual global_size is computed by multiplying the local_size to
        griddim.
        """
        griddim = _ensure_list(args[0])
        blockdim = _ensure_list(args[1])
        size = max(len(griddim), len(blockdim))
        _ensure_size_or_append(griddim, size)
        _ensure_size_or_append(blockdim, size)
        # Compute global_size
        gs = [g * l for g, l in zip(griddim, blockdim)]
        return self.configure(gs, blockdim, *args[2:])


_CacheEntry = namedtuple("_CachedEntry", ['symbol', 'executable',
                                          'kernarg_region'])


class _CachedProgram(object):
    def __init__(self, entry_name, binary):
        self._entry_name = entry_name
        self._binary = binary
        # key: ocl context
        self._cache = {}

    def get(self):
        from .ocldrv import devices

        device = devices.get_device()
        context = devices.get_context()

        result = self._cache.get(context)
        if result is not None:
            program = result[1]
            kernel = result[2]
        else: # The program has not been finalized for this device
            # Build according to the OpenCL version, 2.0 or 2.1
            if device.opencl_version == (2,0):
                spir2_bc = spir2.llvm_to_spir2(self._binary)
                program = context.create_program_from_binary(spir2_bc)
            elif device.opencl_version == (2,1):
                spirv_bc = spirv.llvm_to_spirv(self._binary)
                program = context.create_program_from_il(spirv_bc)

            program.build()
            kernel = program.create_kernel(self._entry_name)

            # Cache the just built cl_program, its cl_device and a cl_kernel
            self._cache[context] = (device,program,kernel)

        return context, device, program, kernel


class OCLKernel(OCLKernelBase):
    """
    A OCL kernel object
    """
    def __init__(self, llvm_module, name, argtypes):
        super(OCLKernel, self).__init__()
        self._llvm_module = llvm_module
        self.assembly = self.binary = llvm_module.__str__()
        self.entry_name = name
        self.argument_types = tuple(argtypes)
        self._argloc = []
        # cached finalized program
        self._cacheprog = _CachedProgram(entry_name=self.entry_name,
                                         binary=self.binary)

    def bind(self):
        """
        Bind kernel to device
        """
        return self._cacheprog.get()


    def __call__(self, *args):
        context, device, program, kernel = self.bind()
        queue = devices.get_queue()

        # Unpack pyobject values into ctypes scalar values
        retr = []  # hold functors for writeback
        kernelargs = []
        for ty, val in zip(self.argument_types, args):
            self._unpack_argument(ty, val, queue, retr, kernelargs)

        # Insert kernel arguments
        kernel.set_args(kernelargs)

        # Invoke kernel
        queue.enqueue_nd_range_kernel(kernel, len(self.global_size),
                                      self.global_size, self.local_size)
        queue.finish()

        # retrieve auto converted arrays
        for wb in retr:
            wb()


    def _unpack_argument(self, ty, val, queue, retr, kernelargs):
        """
        Convert arguments to ctypes and append to kernelargs
        """
        if isinstance(ty, types.Array):
            if isinstance(ty, types.SmartArrayType):
                devary = val.get('gpu')
                retr.append(lambda: val.mark_changed('gpu'))
                outer_parent = ctypes.c_void_p(0)
                kernelargs.append(outer_parent)
            else:
                devary, conv = devicearray.auto_device(val, stream=queue)
                if conv:
                    retr.append(lambda: devary.copy_to_host(val, stream=queue))

            c_intp = ctypes.c_ssize_t

            meminfo = ctypes.c_void_p(0)
            parent = ctypes.c_void_p(0)
            nitems = c_intp(devary.size)
            itemsize = c_intp(devary.dtype.itemsize)
            data = driver.device_pointer(devary) # @@
            kernelargs.append(meminfo)
            kernelargs.append(parent)
            kernelargs.append(nitems)
            kernelargs.append(itemsize)
            kernelargs.append(data)
            for ax in range(devary.ndim):
                kernelargs.append(c_intp(devary.shape[ax]))
            for ax in range(devary.ndim):
                kernelargs.append(c_intp(devary.strides[ax]))

        elif isinstance(ty, types.Integer):
            cval = getattr(ctypes, "c_%s" % ty)(val)
            kernelargs.append(cval)

        elif ty == types.float64:
            cval = ctypes.c_double(val)
            kernelargs.append(cval)

        elif ty == types.float32:
            cval = ctypes.c_float(val)
            kernelargs.append(cval)

        elif ty == types.boolean:
            cval = ctypes.c_uint8(int(val))
            kernelargs.append(cval)

        elif ty == types.complex64:
            kernelargs.append(ctypes.c_float(val.real))
            kernelargs.append(ctypes.c_float(val.imag))

        elif ty == types.complex128:
            kernelargs.append(ctypes.c_double(val.real))
            kernelargs.append(ctypes.c_double(val.imag))

        else:
            raise NotImplementedError(ty, val)



class AutoJitOCLKernel(OCLKernelBase):
    def __init__(self, func):
        super(AutoJitOCLKernel, self).__init__()
        self.py_func = func
        self.definitions = {}

        from .descriptor import OCLTargetDesc

        self.typingctx = OCLTargetDesc.typingctx

    def __call__(self, *args):
        kernel = self.specialize(*args)
        cfg = kernel.configure(self.global_size, self.local_size, self.queue)
        cfg(*args)

    def specialize(self, *args):
        argtypes = tuple([self.typingctx.resolve_argument_type(a)
                          for a in args])
        kernel = self.definitions.get(argtypes)
        if kernel is None:
            kernel = compile_kernel(self.py_func, argtypes)
            self.definitions[argtypes] = kernel
        return kernel

