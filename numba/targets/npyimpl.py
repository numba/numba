from __future__ import print_function, division, absolute_import
import numpy
import math
from llvm.core import Constant, Type
from numba.utils import PYVERSION
from numba import typing, types, cgutils
from numba.targets.imputils import implement
from numba import numpy_support

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
                dres = fnwork(builder, [dval])
                res = context.cast(builder, dres, types.float64, dtype)
            else:
                res = fnwork(builder, [ival])
            builder.store(res, po)

        return out
    return impl


@register
@implement(numpy.absolute, types.Kind(types.Array), types.Kind(types.Array))
def numpy_absolute(context, builder, sig, args):
    imp = numpy_unary_ufunc(types.abs_type)
    return imp(context, builder, sig, args)


@register
@implement(numpy.exp, types.Kind(types.Array), types.Kind(types.Array))
def numpy_exp(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.exp, asfloat=True)
    return imp(context, builder, sig, args)


@register
@implement(numpy.sin, types.Kind(types.Array), types.Kind(types.Array))
def numpy_sin(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.sin, asfloat=True)
    return imp(context, builder, sig, args)


@register
@implement(numpy.cos, types.Kind(types.Array), types.Kind(types.Array))
def numpy_cos(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.cos, asfloat=True)
    return imp(context, builder, sig, args)


@register
@implement(numpy.tan, types.Kind(types.Array), types.Kind(types.Array))
def numpy_tan(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.tan, asfloat=True)
    return imp(context, builder, sig, args)


def numpy_binary_ufunc(core, divbyzero=False):
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
            if divbyzero:
                # Handle division
                iszero = cgutils.is_scalar_zero(builder, y)
                with cgutils.ifelse(builder, iszero, expect=False) as (then,
                                                                       orelse):
                    with then:
                        # Divide by zero
                        if tyout.dtype in types.real_domain:
                            # If x is float and is 0 also, return Nan; else
                            # return Inf
                            shouldretnan = cgutils.is_scalar_zero(builder, x)
                            nan = Constant.real(y.type, float("nan"))
                            inf = Constant.real(y.type, float("inf"))
                            res = builder.select(shouldretnan, nan, inf)
                        elif (tyout.dtype in types.signed_domain and
                                not numpy_support.int_divbyzero_returns_zero):
                            res = Constant.int(y.type, 0x1 << (y.type.width-1))
                        else:
                            res = Constant.null(y.type)
                        builder.store(res, po)
                    with orelse:
                        # Normal
                        res = core(builder, (x, y))
                        builder.store(res, po)
            else:
                # Handle other operations
                res = core(builder, (x, y))
                builder.store(res, po)

        return out
    return impl


@register
@implement(numpy.add, types.Kind(types.Array), types.Kind(types.Array),
           types.Kind(types.Array))
def numpy_add(context, builder, sig, args):
    dtype = sig.args[0].dtype
    coresig = typing.signature(types.Any, dtype, dtype)
    core = context.get_function("+", coresig)
    imp = numpy_binary_ufunc(core)
    return imp(context, builder, sig, args)


@register
@implement(numpy.subtract, types.Kind(types.Array), types.Kind(types.Array),
           types.Kind(types.Array))
def numpy_sub(context, builder, sig, args):
    dtype = sig.args[0].dtype
    coresig = typing.signature(types.Any, dtype, dtype)
    core = context.get_function("-", coresig)
    imp = numpy_binary_ufunc(core)
    return imp(context, builder, sig, args)


@register
@implement(numpy.multiply, types.Kind(types.Array), types.Kind(types.Array),
           types.Kind(types.Array))
def numpy_multiply(context, builder, sig, args):
    dtype = sig.args[0].dtype
    coresig = typing.signature(types.Any, dtype, dtype)
    core = context.get_function("*", coresig)
    imp = numpy_binary_ufunc(core)
    return imp(context, builder, sig, args)


@register
@implement(numpy.divide, types.Kind(types.Array), types.Kind(types.Array),
           types.Kind(types.Array))
def numpy_divide(context, builder, sig, args):
    dtype = sig.args[0].dtype
    isig = typing.signature(dtype, dtype, dtype)
    if dtype in types.signed_domain:
        if PYVERSION >= (3, 0):
            # FIXME This only handles integer output
            # divide in py3 can have real output
            int_sfloordiv_impl = context.get_function("//", isig)
            imp = numpy_binary_ufunc(int_sfloordiv_impl, divbyzero=True)
        else:
            int_sdiv_impl = context.get_function("/?", isig)
            imp = numpy_binary_ufunc(int_sdiv_impl, divbyzero=True)
    elif dtype in types.unsigned_domain:
        int_udiv_impl = context.get_function("/?", isig)
        imp = numpy_binary_ufunc(int_udiv_impl, divbyzero=True)
    else:
        real_div_impl = context.get_function("/?", isig)
        imp = numpy_binary_ufunc(real_div_impl, divbyzero=True)

    return imp(context, builder, sig, args)
