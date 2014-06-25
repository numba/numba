from __future__ import print_function, absolute_import
import copy
import ctypes
from numba import types, compiler


def compile_ocl(pyfunc, return_type, args, debug):
    # First compilation will trigger the initialization of the CUDA backend.
    from .descriptor import OCLTargetDesc

    typingctx = OCLTargetDesc.typingctx
    targetctx = OCLTargetDesc.targetctx
    # TODO handle debug flag
    flags = compiler.Flags()
    # Do not compile (generate native code), just lower (to LLVM)
    flags.set('no_compile')
    # Run compilation pipeline
    cres = compiler.compile_extra(typingctx=typingctx,
                                  targetctx=targetctx,
                                  func=pyfunc,
                                  args=args,
                                  return_type=return_type,
                                  flags=flags,
                                  locals={})

    # Fix global naming
    for gv in cres.llvm_module.global_variables:
        if '.' in gv.name:
            gv.name = gv.name.replace('.', '_')

    return cres


def compile_kernel(pyfunc, args, debug=False):
    cres = compile_ocl(pyfunc, types.void, args, debug=debug)
    kernel = cres.target_context.prepare_ocl_kernel(cres.llvm_func,
                                                    cres.signature.args)
    cres = cres._replace(llvm_func=kernel)

    kern = OCLKernel(llvm_module=cres.llvm_module,
                     name=cres.llvm_func.name,
                     argtypes=cres.signature.args)
    return kern


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

    def configure(self, global_size, local_size, stream=None):
        global_size = _ensure_list(global_size)
        local_size = _ensure_list(local_size)

        size = max(len(global_size), len(local_size))

        _ensure_size_or_append(global_size, size)
        _ensure_size_or_append(local_size, size)

        clone = self.copy()
        clone.global_size = tuple(global_size)
        clone.local_size = tuple(local_size)
        clone.stream = stream

        return clone

    def __getitem__(self, args):
        return self.configure(*args)


class OCLKernel(OCLKernelBase):
    """
    A OpenCL kernel object
    """
    def __init__(self, llvm_module, name, argtypes):
        super(OCLKernel, self).__init__()
        self.llvm_module = llvm_module
        self.binary = self.llvm_module.to_bitcode()
        self.entry_name = name
        self.argument_types = tuple(argtypes)

        self.device, self.program, self.kernel = self.bind()

    def bind(self):
        """
        Bind kernel to device
        """
        from .ocldrv import cl
        from .target import SPIR_BUILD_OPTION

        device = cl.default_platform.default_device
        context = cl.create_context(device.platform, [device])
        program = context.create_program_from_binary(self.binary)
        program.build(options=SPIR_BUILD_OPTION)

        kernel = program.create_kernel(self.entry_name.encode('utf8'))

        return device, program, kernel

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

    raise NotImplementedError(ty)


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









