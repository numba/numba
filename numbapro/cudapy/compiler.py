from __future__ import absolute_import, print_function

import llvm.core as lc

from numba import compiler, types
from numbapro.cudadrv import nvvm, devices
from .execution import CUDAKernel
from . import ptxlib, libdevice, ptx
from numbapro.cudabackend.target import CUDATargetDesc


def _set_flags(debug):
    flags = list(compiler.DEFAULT_FLAGS)
    if not debug:
        flags.append('no-exceptions')
    return flags


def compile_kernel(func, argtys, debug=False):
    lmod, lfunc, excs = compile_common(func, types.void, argtys,
                                       flags=_set_flags(debug))
    # PTX-ization
    wrapper = generate_kernel_wrapper(lfunc, bool(excs))
    cudakernel = CUDAKernel(wrapper.name, to_ptx(wrapper), argtys, excs)
    return cudakernel


def generate_kernel_wrapper(lfunc, has_excs):
    fname = '_cudapy_wrapper_' + lfunc.name
    argtypes = list(lfunc.type.pointee.args)
    if has_excs:
        exctype = lc.Type.int()
        argtypes.append(lc.Type.pointer(exctype)) # for exception
    fntype = lc.Type.function(lc.Type.void(), argtypes)

    wrapper = lfunc.module.add_function(fntype, fname)

    builder = lc.Builder.new(wrapper.append_basic_block(''))

    if has_excs:
        exc = builder.call(lfunc, wrapper.args[:-1])
    else:
        exc = builder.call(lfunc, wrapper.args)

    if has_excs:
        no_exc = lc.Constant.null(exctype)
        raised = builder.icmp(lc.ICMP_NE, exc, no_exc)

        with cgutils.if_then(builder, raised):
            fname_tx = ptx.SREG_MAPPING[ptx._ptx_sreg_tidx]
            fname_ty = ptx.SREG_MAPPING[ptx._ptx_sreg_tidy]
            fname_tz = ptx.SREG_MAPPING[ptx._ptx_sreg_tidz]

            fname_bx = ptx.SREG_MAPPING[ptx._ptx_sreg_ctaidx]
            fname_by = ptx.SREG_MAPPING[ptx._ptx_sreg_ctaidy]

            li32 = types.uint32.llvm_as_value()
            fn_tx = cgutils.get_function(builder, fname_tx, li32, ())
            fn_ty = cgutils.get_function(builder, fname_ty, li32, ())
            fn_tz = cgutils.get_function(builder, fname_tz, li32, ())

            fn_bx = cgutils.get_function(builder, fname_bx, li32, ())
            fn_by = cgutils.get_function(builder, fname_by, li32, ())

            tx = builder.call(fn_tx, ())
            ty = builder.call(fn_ty, ())
            tz = builder.call(fn_tz, ())

            bx = builder.call(fn_bx, ())
            by = builder.call(fn_by, ())

            excptr = wrapper.args[-1]

            old = builder.atomic_cmpxchg(excptr, no_exc, exc, 'monotonic')

            success = builder.icmp(lc.ICMP_EQ, old, no_exc)
            with cgutils.if_then(builder, success):
                values = [tx, ty, tz, bx, by]
                for i, val in enumerate(values):
                    offset = types.uint32.llvm_const(1 + i)
                    ptr = builder.gep(wrapper.args[-1], [offset])
                    builder.store(val, ptr)

                builder.ret_void()

            builder.ret_void()

    builder.ret_void()

    lfunc.add_attribute(lc.ATTR_ALWAYS_INLINE)
    return wrapper

#
# def compile_device(func, retty, argtys, inline=False, debug=False):
#     lmod, lfunc, excs = compile_common(func, retty, argtys,
#                                        flags=_set_flags(debug))
#     if inline:
#         lfunc.add_attribute(lc.ATTR_ALWAYS_INLINE)
#     return DeviceFunction(func, lmod, lfunc, retty, argtys, excs)


def declare_device_function(name, retty, argtys):
    lmod = lc.Module.new('extern-%s' % name)
    lret = retty.llvm_as_return()
    largs = [t.llvm_as_argument() for t in argtys]
    lfty = lc.Type.function(lc.Type.void(), largs + [lret])
    lfunc = lmod.add_function(lfty, name=name)
    edf = ExternalDeviceFunction(name, lmod, lfunc, retty, argtys)
    return edf


class ExternalDeviceFunction(object):
    def __init__(self, name, lmod, lfunc, retty, argtys):
        self.name = name
        self.args = tuple(argtys)
        self.return_type = retty
        self._npm_context_ = lmod, lfunc, self.return_type, self.args, None

    def __repr__(self):
        args = (self.name, self.return_type or 'void', self.args)
        return '<cuda external device function %s %s%s>' % args


######


def compile_cuda(pyfunc, return_type, args, debug):
    typingctx = CUDATargetDesc.typingctx
    targetctx = CUDATargetDesc.targetctx
    # TODO handle debug flag
    flags = compiler.Flags()
    # Do not compile, just lower
    flags.set('no_compile')
    # Run compilation pipeline
    cres = compiler.compile_extra(typingctx=typingctx,
                                  targetctx=targetctx,
                                  func=pyfunc,
                                  args=args,
                                  return_type=return_type,
                                  flags=flags,
                                  locals={})
    return cres


def compile_kernel(pyfunc, args, link, debug=False):
    cres = compile_cuda(pyfunc, types.void, args, debug=debug)
    kernel = cres.targetctx.prepare_cuda_kernel(cres.llvm_func,
                                                cres.signature.args)
    cres = cres._replace(llvm_func=kernel)
    cukern = CUDAKernel(llvm_module=cres.llvm_module,
                        name=cres.llvm_func.name,
                        argtypes=cres.signature.args,
                        link=link)
    return cukern


def compile_device(pyfunc, return_type, args, inline=True, debug=False):
    cres = compile_cuda(pyfunc, return_type, args, debug=debug)
    return DeviceFunction(cres)


class DeviceFunction(object):
    def __init__(self, cres):
        self.cres = cres



