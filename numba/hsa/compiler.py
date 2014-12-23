from __future__ import print_function, absolute_import
import copy
import ctypes
from numba.typing.templates import ConcreteTemplate
from numba import types, compiler
from .hlc import hlc


def compile_hsa(pyfunc, return_type, args, debug, functype):
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
    cres = compile_hsa(pyfunc, types.void, args, debug=debug, functype='kernel')
    func = cres.library.get_function(cres.fndesc.llvm_func_name)
    kernel = cres.target_context.prepare_hsa_kernel(func, cres.signature.args)
    hsakern = HSAKernel(llvm_module=cres.library._final_module,
                        name=kernel.name,
                        argtypes=cres.signature.args)
    return hsakern


def compile_device(pyfunc, return_type, args, debug=False):
    cres = compile_hsa(pyfunc, return_type, args, debug=debug)
    cres.target_context.mark_ocl_device(cres.llvm_func)
    devfn = DeviceFunction(cres)

    class device_function_template(ConcreteTemplate):
        key = devfn
        cases = [cres.signature]

    cres.typing_context.insert_user_function(devfn, device_function_template)
    libs = [cres.llvm_module]
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
    A OpenCL kernel object
    """

    def __init__(self, llvm_module, name, argtypes):
        super(HSAKernel, self).__init__()
        self.llvm_module = llvm_module

        hlcmod = hlc.Module()
        for m in llvm_module._modules:
            if m.global_variables:  # XXX: ignore empty module for now
                hlcmod.load_llvm(str(m))
        hsail = hlcmod.finalize()
        print(hsail)
        raise NotImplementedError
        self.binary = hsail
        self.entry_name = name
        self.argument_types = tuple(argtypes)

        self.device, self.program, self.kernel = self.bind()

    def bind(self):
        """
        Bind kernel to device
        """
        raise NotImplementedError

    def __call__(self, *args):
        # Prepare arguments
        iter_unboxed = _prepare_arguments(args, self.argument_types)
        for i, argval in enumerate(iter_unboxed):
            self.kernel.set_arg(i, argval)

        # Invoke kernel
        do_sync = False
        queue = self.stream
        if queue is None:
            do_sync = True
            queue = self.program.context.create_command_queue(self.device)

        queue.enqueue_nd_range_kernel(self.kernel, len(self.global_size),
                                      self.global_size, self.local_size)

        if do_sync:
            queue.finish()


def _prepare_arguments(args, argtys):
    for val, ty in zip(args, argtys):
        for x in _unbox(val, ty):
            yield x


def _unbox(val, ty):
    if isinstance(ty, types.Array):
        data = val.data
        shapes = [_unbox(s, types.intp) for s in val.shape]
        strides = [_unbox(s, types.intp) for s in val.strides]
        return [data] + shapes + strides

    elif ty in types.integer_domain:
        return INTEGER_TYPE_MAP[ty](val)

    elif ty in types.real_domain:
        return REAL_TYPE_MAP[ty](val)

    elif ty in types.complex_domain:
        return COMPLEX_TYPE_MAP[ty](val)

    raise NotImplementedError(ty)


def make_complex_ctypes(element):
    class BaseComplex(ctypes.Structure):
        _fields = [
            ('real', element),
            ('imag', element),
        ]

        def __init__(self, real_or_cmpl=0, imag=0):
            if isinstance(real_or_cmpl, float):
                self.real = real_or_cmpl
                self.imag = imag
            else:
                self.real = real_or_cmpl.real
                self.imag = real_or_cmpl.imag

    return BaseComplex


Complex = make_complex_ctypes(ctypes.c_float)
DoubleComplex = make_complex_ctypes(ctypes.c_double)

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



