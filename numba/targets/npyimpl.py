from __future__ import print_function, division, absolute_import
import numpy
import math
from numba.utils import PYVERSION
from numba import typing, types, cgutils
from numba.targets.imputils import implement

functions = []


def register(f):
    functions.append(f)
    return f


def numpy_unary_ufunc(funckey, asfloat=False):
    def impl(context, builder, sig, args):
        [tyinp, tyout] = sig.args
        [inp, out] = args
        assert tyinp.dtype == tyout.dtype
        dtype = tyinp.dtype
        ndim = tyinp.ndim

        iary = context.make_array(tyinp)(context, builder, inp)
        oary = context.make_array(tyout)(context, builder, out)

        if asfloat:
            sig = typing.signature(types.float64, types.float64)
        else:
            sig = typing.signature(dtype, dtype)

        fnwork = context.get_function(funckey, sig)
        intpty = context.get_value_type(types.intp)

        # TODO handle differing shape by mimicking broadcasting
        shape = cgutils.unpack_tuple(builder, iary.shape, ndim)
        with cgutils.loop_nest(builder, shape, intp=intpty) as indices:
            pi = cgutils.get_item_pointer(builder, tyinp, iary, indices)
            po = cgutils.get_item_pointer(builder, tyout, oary, indices)

            ival = builder.load(pi)
            if asfloat:
                dval = context.cast(builder, ival, dtype, types.float64)
                fnwork_sig = typing.signature(types.float64, types.float64)
                dres = fnwork(context, builder, fnwork_sig, [dval])
                res = context.cast(builder, dres, types.float64, dtype)
            else:
                fnwork_sig = typing.signature(dtype, dtype)
                res = fnwork(context, builder, fnwork_sig, [ival])
            builder.store(res, po)

        return out
    return impl


@register
@implement(numpy.absolute, types.Kind(types.Array), types.Kind(types.Array),
           types.Kind(types.Array))
def numpy_absolute(context, builder, sig, args):
    imp = numpy_unary_ufunc(types.abs_type)
    return imp(context, builder, sig, args)


@register
@implement(numpy.exp, types.Kind(types.Array), types.Kind(types.Array),
           types.Kind(types.Array))
def numpy_exp(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.exp, asfloat=True)
    return imp(context, builder, sig, args)


@register
@implement(numpy.sin, types.Kind(types.Array), types.Kind(types.Array),
           types.Kind(types.Array))
def numpy_sin(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.sin, asfloat=True)
    return imp(context, builder, sig, args)


@register
@implement(numpy.cos, types.Kind(types.Array), types.Kind(types.Array),
           types.Kind(types.Array))
def numpy_cos(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.cos, asfloat=True)
    return imp(context, builder, sig, args)


@register
@implement(numpy.tan, types.Kind(types.Array), types.Kind(types.Array),
           types.Kind(types.Array))
def numpy_tan(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.tan, asfloat=True)
    return imp(context, builder, sig, args)


def numpy_binary_ufunc(core):
    def impl(context, builder, sig, args):
        [tyvx, tywy, tyout] = sig.args
        [vx, wy, out] = args
        assert tyvx.dtype == tyout.dtype
        assert tyvx.dtype == tywy.dtype
        dtype = tyvx.dtype
        ndim = tyvx.ndim

        xary = context.make_array(tyvx)(context, builder, vx)
        yary = context.make_array(tywy)(context, builder, wy)
        oary = context.make_array(tyout)(context, builder, out)

        intpty = context.get_value_type(types.intp)

        # TODO handle differing shape by mimicking broadcasting
        loopshape = cgutils.unpack_tuple(builder, xary.shape, ndim)

        xyo_shape = [cgutils.unpack_tuple(builder, ary.shape, ndim)
                     for ary in (xary, yary, oary)]
        xyo_strides = [cgutils.unpack_tuple(builder, ary.strides, ndim)
                       for ary in (xary, yary, oary)]
        xyo_data = [ary.data for ary in (xary, yary, oary)]
        xyo_layout = [ty.layout for ty in (tyvx, tywy, tyout)]

        with cgutils.loop_nest(builder, loopshape, intp=intpty) as indices:
            [px, py, po] = [cgutils.get_item_pointer2(builder,
                                                      data=data, shape=shape,
                                                      strides=strides,
                                                      layout=layout,
                                                      inds=indices)
                            for data, shape, strides, layout
                            in zip(xyo_data, xyo_shape, xyo_strides,
                                   xyo_layout)]

            x = builder.load(px)
            y = builder.load(py)
            res = core(context, builder, (dtype, dtype), (x, y))
            builder.store(res, po)

        return out
    return impl


@register
@implement(numpy.add, types.Kind(types.Array), types.Kind(types.Array),
           types.Kind(types.Array), types.Kind(types.Array))
def numpy_add(context, builder, sig, args):
    def intcore(context, builder, sig, args):
        x, y = args
        return builder.add(x, y)

    def realcore(context, builder, sig, args):
        x, y = args
        return builder.fadd(x, y)

    if sig.args[0].dtype in types.integer_domain:
        imp = numpy_binary_ufunc(intcore)
    else:
        imp = numpy_binary_ufunc(realcore)

    return imp(context, builder, sig, args)


@register
@implement(numpy.subtract, types.Kind(types.Array), types.Kind(types.Array),
           types.Kind(types.Array), types.Kind(types.Array))
def numpy_sub(context, builder, sig, args):
    def intcore(context, builder, sig, args):
        x, y = args
        return builder.sub(x, y)

    def realcore(context, builder, sig, args):
        x, y = args
        return builder.fsub(x, y)

    if sig.args[0].dtype in types.integer_domain:
        imp = numpy_binary_ufunc(intcore)
    else:
        imp = numpy_binary_ufunc(realcore)

    return imp(context, builder, sig, args)


@register
@implement(numpy.multiply, types.Kind(types.Array), types.Kind(types.Array),
           types.Kind(types.Array), types.Kind(types.Array))
def numpy_multiply(context, builder, sig, args):
    def intcore(context, builder, sig, args):
        x, y = args
        return builder.mul(x, y)

    def realcore(context, builder, sig, args):
        x, y = args
        return builder.fmul(x, y)

    if sig.args[0].dtype in types.integer_domain:
        imp = numpy_binary_ufunc(intcore)
    else:
        imp = numpy_binary_ufunc(realcore)

    return imp(context, builder, sig, args)


@register
@implement(numpy.divide, types.Kind(types.Array), types.Kind(types.Array),
           types.Kind(types.Array), types.Kind(types.Array))
def numpy_divide(context, builder, sig, args):
    dtype = sig.args[0].dtype
    isig = typing.signature(dtype, dtype, dtype)
    if dtype in types.signed_domain:
        if PYVERSION >= (3, 0):
            int_sfloordiv_impl = context.get_function("//", isig)
            imp = numpy_binary_ufunc(int_sfloordiv_impl)
        else:
            int_sdiv_impl = context.get_function("/?", isig)
            imp = numpy_binary_ufunc(int_sdiv_impl)
    elif dtype in types.unsigned_domain:
        int_udiv_impl = context.get_function("/?", isig)
        imp = numpy_binary_ufunc(int_udiv_impl)
    else:
        real_div_impl = context.get_function("/?", isig)
        imp = numpy_binary_ufunc(real_div_impl)

    return imp(context, builder, sig, args)
