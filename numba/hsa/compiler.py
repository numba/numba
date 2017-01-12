from __future__ import print_function, absolute_import
import copy
from collections import namedtuple

from numba.typing.templates import ConcreteTemplate
from numba import types, compiler
from .hlc import hlc
from .hsadrv import devices, driver
from numba.typing.templates import AbstractTemplate
from numba import ctypes_support as ctypes

def compile_hsa(pyfunc, return_type, args, debug):
    # First compilation will trigger the initialization of the HSA backend.
    from .descriptor import HSATargetDesc

    typingctx = HSATargetDesc.typingctx
    targetctx = HSATargetDesc.targetctx
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
    cres = compile_hsa(pyfunc, types.void, args, debug=debug)
    func = cres.library.get_function(cres.fndesc.llvm_func_name)
    kernel = cres.target_context.prepare_hsa_kernel(func, cres.signature.args)
    hsakern = HSAKernel(llvm_module=kernel.module,
                        name=kernel.name,
                        argtypes=cres.signature.args)
    return hsakern


def compile_device(pyfunc, return_type, args, debug=False):
    cres = compile_hsa(pyfunc, return_type, args, debug=debug)
    func = cres.library.get_function(cres.fndesc.llvm_func_name)
    cres.target_context.mark_hsa_device(func)
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
    from .descriptor import HSATargetDesc

    dft = DeviceFunctionTemplate(pyfunc)

    class device_function_template(AbstractTemplate):
        key = dft

        def generic(self, args, kws):
            assert not kws
            return dft.compile(args)

    typingctx = HSATargetDesc.typingctx
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
            cres = compile_hsa(self.py_func, None, args, debug=self.debug)
            func = cres.library.get_function(cres.fndesc.llvm_func_name)
            cres.target_context.mark_hsa_device(func)
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


class HSAKernelBase(object):
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
        # key: hsa context
        self._cache = {}

    def get(self):
        ctx = devices.get_context()
        result = self._cache.get(ctx)
        # The program has not been finalized for this device
        if result is None:
            # Finalize
            symbol = '&{0}'.format(self._entry_name)
            agent = ctx.agent

            brig_module = driver.BrigModule(self._binary)
            prog = driver.Program()
            prog.add_module(brig_module)
            code = prog.finalize(agent.isa)
            del prog

            ex = driver.Executable()
            ex.load(agent, code)
            ex.freeze()
            symobj = ex.get_symbol(agent, symbol)
            kernarg_region = [r for r in agent.regions.globals
                              if r.supports_kernargs][0]

            # Cache the finalized program
            result = _CacheEntry(symbol=symobj, executable=ex,
                                 kernarg_region=kernarg_region)
            self._cache[ctx] = result

        return ctx, result


class HSAKernel(HSAKernelBase):
    """
    A HSA kernel object
    """
    def __init__(self, llvm_module, name, argtypes):
        super(HSAKernel, self).__init__()
        self._llvm_module = llvm_module
        self.assembly, self.binary = self._finalize()
        self.entry_name = name
        self.argument_types = tuple(argtypes)
        self._argloc = []
        # cached finalized program
        self._cacheprog = _CachedProgram(entry_name=self.entry_name,
                                         binary=self.binary)

    def _finalize(self):
        hlcmod = hlc.Module()
        hlcmod.load_llvm(str(self._llvm_module))
        return hlcmod.finalize()

    def bind(self):
        """
        Bind kernel to device
        """
        ctx, entry = self._cacheprog.get()
        if entry.symbol.kernarg_segment_size > 0:
            kernarg_type = (ctypes.c_byte * entry.symbol.kernarg_segment_size)
            kernargs = entry.kernarg_region.allocate(kernarg_type)
        else:
            kernargs = None
        return ctx, entry.symbol, kernargs, entry.kernarg_region

    def __call__(self, *args):
        ctx, symbol, kernargs, kernarg_region = self.bind()

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

        assert (kernargs is None or
                base <= ctypes.sizeof(kernargs)), "Kernel argument size is invalid"

        # Actual Kernel launch
        qq = ctx.default_queue

        # Dispatch
        qq.dispatch(symbol, kernargs, workgroup_size=self.local_size,
                    grid_size=self.global_size)

        # Free kernel region
        if kernargs is not None:
            kernarg_region.free(kernargs)


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


class AutoJitHSAKernel(HSAKernelBase):
    def __init__(self, func):
        super(AutoJitHSAKernel, self).__init__()
        self.py_func = func
        self.definitions = {}

        from .descriptor import HSATargetDesc

        self.typingctx = HSATargetDesc.typingctx

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

