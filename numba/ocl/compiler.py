from __future__ import print_function, absolute_import
import copy
from collections import namedtuple

from numba.typing.templates import ConcreteTemplate
from numba import types, compiler
from .ocldrv import devices, driver
from numba.typing.templates import AbstractTemplate
from numba import ctypes_support as ctypes
from subprocess import Popen, PIPE, STDOUT


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
        self.stream = None

    def copy(self):
        return copy.copy(self)

    def configure(self, global_size, local_size=None, stream=None):
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
        clone.stream = stream

        return clone

    def forall(self, nelem, local_size=64, stream=None):
        """Simplified configuration for 1D kernel launch
        """
        return self.configure(nelem, min(nelem, local_size), stream=stream)

    def __getitem__(self, args):
        """Mimick CUDA python's square-bracket notation for configuration.
        This assumes a the argument to be:
            `griddim, blockdim, stream`
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
        from .ocldrv.driver import driver

        ctx = devices.get_context()
        result = self._cache.get(ctx)
        # The program has not been finalized for this device
        if result is None:
            # Finalize the building
            device = driver.default_platform.default_device
            context = driver.create_context(device.platform, [device])
            #program = context.create_program_from_binary(self._binary)
            #program.build(options=b"-x spir -spir-std=2.0")

            llc_arg = "/home/jcaraban/jesus/code/spirv/bin/llc -march=spirv64 -o -".split()
            llc_pro = Popen(llc_arg, stdin=PIPE, stdout=PIPE)
            llc_out = llc_pro.communicate(input=self._binary)[0]
            spirv = llc_out#.decode()
            program = context.create_program_from_il(spirv)
            program.build()

            kernel = program.create_kernel(self._entry_name)

            # Cache the just built cl_program, its cl_device and a cl_kernel
            self._cache[context] = (device,progra,kernel)

        return device, program, kernel


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

        # Unpack pyobject values into ctypes scalar values
        expanded_values = []
        for ty, val in zip(self.argument_types, args):
            _unpack_argument(ty, val, expanded_values)

        # Insert kernel arguments
        base = 0
        for av in expanded_values:
            # Adjust for alignemnt
            align = ctypes.sizeof(av)
            pad = _calc_padding_for_alignment(align, base)
            base += pad
            # Move to offset
            offseted = ctypes.addressof(kernargs) + base
            asptr = ctypes.cast(offseted, ctypes.POINTER(type(av)))
            # Assign value
            asptr[0] = av
            # Increment offset
            base += align

        # Invoke kernel
        do_sync = False
        queue = self.stream
        if queue is None:
            do_sync = True
            queue = program.context.create_command_queue(device)

        queue.enqueue_nd_range_kernel(self.kernel, len(self.global_size),
                                      self.global_size, self.local_size)

        if do_sync:
            queue.finish()


def _unpack_argument(ty, val, kernelargs):
    """
    Convert arguments to ctypes and append to kernelargs
    """
    if isinstance(ty, types.Array):
        c_intp = ctypes.c_ssize_t

        meminfo = parent = ctypes.c_void_p(0)
        nitems = c_intp(val.size)
        itemsize = c_intp(val.dtype.itemsize)
        data = ctypes.c_void_p(val.ctypes.data)
        kernelargs.append(meminfo)
        kernelargs.append(parent)
        kernelargs.append(nitems)
        kernelargs.append(itemsize)
        kernelargs.append(data)
        for ax in range(val.ndim):
            kernelargs.append(c_intp(val.shape[ax]))
        for ax in range(val.ndim):
            kernelargs.append(c_intp(val.strides[ax]))

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


def _calc_padding_for_alignment(align, base):
    """
    Returns byte padding required to move the base pointer into proper alignment
    """
    rmdr = int(base) % align
    if rmdr == 0:
        return 0
    else:
        return align - rmdr


class AutoJitOCLKernel(OCLKernelBase):
    def __init__(self, func):
        super(AutoJitOCLKernel, self).__init__()
        self.py_func = func
        self.definitions = {}

        from .descriptor import OCLTargetDesc

        self.typingctx = OCLTargetDesc.typingctx

    def __call__(self, *args):
        kernel = self.specialize(*args)
        cfg = kernel.configure(self.global_size, self.local_size, self.stream)
        cfg(*args)

    def specialize(self, *args):
        argtypes = tuple([self.typingctx.resolve_argument_type(a)
                          for a in args])
        kernel = self.definitions.get(argtypes)
        if kernel is None:
            kernel = compile_kernel(self.py_func, argtypes)
            self.definitions[argtypes] = kernel
        return kernel

