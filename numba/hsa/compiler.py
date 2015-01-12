from __future__ import print_function, absolute_import
import copy
import ctypes

from numba.typing.templates import ConcreteTemplate
from numba import types, compiler
from .hlc import hlc
from .hsadrv import devices, driver
from numba.targets.arrayobj import make_array_ctype


def compile_hsa(pyfunc, return_type, args, debug):
    # First compilation will trigger the initialization of the CUDA backend.
    from .descriptor import HSATargetDesc

    typingctx = HSATargetDesc.typingctx
    targetctx = HSATargetDesc.targetctx
    # TODO handle debug flag
    flags = compiler.Flags()
    # Do not compile (generate native code), just lower (to LLVM)
    flags.set('no_compile')
    flags.set('no_cpython_wrapper')
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
    hsakern = HSAKernel(llvm_module=cres.library._final_module,
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
        self.local_size = None
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


class HSAKernel(HSAKernelBase):
    """
    A HSA kernel object
    """
    INJECTED_NARG = 6

    def __init__(self, llvm_module, name, argtypes):
        super(HSAKernel, self).__init__()
        self._llvm_module = llvm_module
        self.assembly, self.binary = self._finalize()
        self.entry_name = name
        self.argument_types = tuple(argtypes)
        self._arginfos = [ArgumentInfo(argty) for argty in self.argument_types]
        self._argloc = []
        # Calculate argument position
        self._injectedargsize = self.INJECTED_NARG * ctypes.sizeof(
            ctypes.c_void_p)
        kas = 0
        for ai in self._arginfos:
            padding = ai.calc_padding(kas)
            self._argloc.append(padding + kas + self._injectedargsize)
            kas += padding + ai.size
        self._kernargsize = kas + self._injectedargsize

    def _finalize(self):
        hlcmod = hlc.Module()
        for m in self._llvm_module._modules:
            hlcmod.load_llvm(str(m))
        return hlcmod.finalize()

    def bind(self):
        """
        Bind kernel to device
        """
        ctx = devices.get_context()
        symbol = '&{0}'.format(self.entry_name)
        brig_module = driver.BrigModule.from_memory(self.binary)
        symbol_offset = brig_module.find_symbol_offset(symbol)
        agent = ctx.agent
        program = driver.hsa.create_program([agent])
        module = program.add_module(brig_module)
        code_desc = program.finalize(agent, module, symbol_offset)
        kernarg_region = [r for r in agent.regions if r.supports_kernargs][0]
        kernarg_types = (ctypes.c_byte * self._kernargsize)
        kernargs = kernarg_region.allocate(kernarg_types)
        # Inject dummy argument
        injectargs = ctypes.cast(kernargs,
                                 ctypes.POINTER(ctypes.c_void_p *
                                                self.INJECTED_NARG)).contents
        for i in range(self.INJECTED_NARG):
            injectargs[i] = 0

        return ctx, code_desc, kernargs, kernarg_region, program

    def __call__(self, *args):
        ctx, code_desc, kernargs, kernarg_region, program = self.bind()

        # Insert kernel arguments
        keepalive = []
        for byteoffset, arginfo, argval in zip(self._argloc, self._arginfos,
                                               args):
            keepalive.append(arginfo.assign(kernargs, byteoffset, argval))

        qq = ctx.default_queue

        # Dispatch
        qq.dispatch(code_desc, kernargs, workgroup_size=self.local_size,
                    grid_size=self.global_size)

        # Free kernel region
        kernarg_region.free(kernargs)


class ArgumentInfo(object):
    def __init__(self, ty):
        self.byref = False
        if isinstance(ty, types.Array):
            self._arytype = make_array_ctype(ndim=ty.ndim)
            self.ctype = ctypes.c_void_p
            self.byref = True

            def ctor(val):
                cval = self._arytype(parent=None,
                                     data=val.ctypes.data,
                                     shape=val.ctypes.shape,
                                     strides=val.ctypes.strides)
                return cval

            self.ctor = ctor

        elif ty in REAL_TYPE_MAP:
            self.ctype = self.ctor = REAL_TYPE_MAP[ty]

        elif ty in INTEGER_TYPE_MAP:
            self.ctype = self.ctor = INTEGER_TYPE_MAP[ty]

        elif ty in COMPLEX_TYPE_MAP:
            self.byref = True
            self.ctype = ctypes.c_void_p
            self.ctor = COMPLEX_TYPE_MAP[ty]

        else:
            raise TypeError(ty)

        self.align = ctypes.sizeof(self.ctype)
        self.size = self.align

    def calc_padding(self, base):
        rmdr = int(base) % self.align
        if rmdr == 0:
            return 0
        else:
            return self.align - rmdr

    def assign(self, bytestorage, offset, val):
        keepalive = cval = self.ctor(val)
        if self.byref:
            cval = ctypes.c_void_p(ctypes.addressof(cval))
        ptr = ctypes.c_void_p(ctypes.addressof(bytestorage) + offset)
        casted = ctypes.cast(ptr, ctypes.POINTER(self.ctype))
        casted[0] = cval
        return keepalive


class _BaseComplex(ctypes.Structure):
    def __init__(self, real_or_cmpl=0, imag=0):
        if isinstance(real_or_cmpl, float):
            self.real = real_or_cmpl
            self.imag = imag
        else:
            self.real = real_or_cmpl.real
            self.imag = real_or_cmpl.imag


class Complex(_BaseComplex):
    _fields_ = [
        ('real', ctypes.c_float),
        ('imag', ctypes.c_float),
    ]


class DoubleComplex(_BaseComplex):
    _fields_ = [
        ('real', ctypes.c_double),
        ('imag', ctypes.c_double),
    ]


INTEGER_TYPE_MAP = {
    types.int8: ctypes.c_int8,
    types.int16: ctypes.c_int16,
    types.int32: ctypes.c_int32,
    types.int64: ctypes.c_int64,
    types.uint8: ctypes.c_uint8,
    types.uint16: ctypes.c_uint16,
    types.uint32: ctypes.c_uint32,
    types.uint64: ctypes.c_uint64,
}

REAL_TYPE_MAP = {
    types.float32: ctypes.c_float,
    types.float64: ctypes.c_double,
}

COMPLEX_TYPE_MAP = {
    types.complex64: Complex,
    types.complex128: DoubleComplex,
}



