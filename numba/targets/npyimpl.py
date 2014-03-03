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


def numpy_unary_ufunc(funckey, asfloat=False, scalar_input=False):
    def impl(context, builder, sig, args):
        [tyinp, tyout] = sig.args
        [inp, out] = args
        if scalar_input:
            ndim = 1
        else:
            ndim = tyinp.ndim

        if not scalar_input:
            iary = context.make_array(tyinp)(context, builder, inp)
        oary = context.make_array(tyout)(context, builder, out)

        if asfloat:
            sig = typing.signature(types.float64, types.float64)
        else:
            if scalar_input:
                sig = typing.signature(tyout.dtype, tyinp)
            else:
                sig = typing.signature(tyout.dtype, tyinp.dtype)

        fnwork = context.get_function(funckey, sig)
        intpty = context.get_value_type(types.intp)

        # TODO handle differing shape by mimicking broadcasting
        if scalar_input:
            shape = cgutils.unpack_tuple(builder, oary.shape, ndim)
        else:
            shape = cgutils.unpack_tuple(builder, iary.shape, ndim)
        with cgutils.loop_nest(builder, shape, intp=intpty) as indices:
            if not scalar_input:
                pi = cgutils.get_item_pointer(builder, tyinp, iary, indices)
            po = cgutils.get_item_pointer(builder, tyout, oary, indices)

            if scalar_input:
                ival = inp
            else:
                ival = builder.load(pi)
            if asfloat:
                if scalar_input:
                    dval = context.cast(builder, ival, tyinp, types.float64)
                else:
                    dval = context.cast(builder, ival, tyinp.dtype, types.float64)
                dres = fnwork(builder, [dval])
                res = context.cast(builder, dres, types.float64, tyout.dtype)
            elif scalar_input and tyinp != tyout:
                tempres = fnwork(builder, [ival])
                res = context.cast(builder, tempres, tyinp, tyout.dtype)
            elif tyinp.dtype != tyout.dtype:
                tempres = fnwork(builder, [ival])
                res = context.cast(builder, tempres, tyinp.dtype, tyout.dtype)
            else:
                res = fnwork(builder, [ival])
            builder.store(res, po)

        return out
    return impl


def numpy_scalar_unary_ufunc(funckey, asfloat=True):
    def impl(context, builder, sig, args):
        [tyinp] = sig.args
        tyout = sig.return_type
        [inp] = args
            
        if asfloat:
            sig = typing.signature(types.float64, types.float64)
        
        fnwork = context.get_function(funckey, sig)
        if asfloat:
            inp = context.cast(builder, inp, tyinp, types.float64)
        res = fnwork(builder, [inp])
        if asfloat:
            res = context.cast(builder, res, types.float64, tyout)
        return res

    return impl


@register
@implement(numpy.absolute, types.Kind(types.Array), types.Kind(types.Array))
def numpy_absolute(context, builder, sig, args):
    imp = numpy_unary_ufunc(types.abs_type)
    return imp(context, builder, sig, args)

def numpy_absolute_scalar_input(context, builder, sig, args):
    imp = numpy_unary_ufunc(types.abs_type, scalar_input=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.absolute, ty, types.Kind(types.Array))(numpy_absolute_scalar_input))

def numpy_absolute_scalar(context, builder, sig, args):
    imp = numpy_scalar_unary_ufunc(types.abs_type)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.absolute, ty)(numpy_absolute_scalar))


@register
@implement(numpy.exp, types.Kind(types.Array), types.Kind(types.Array))
def numpy_exp(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.exp, asfloat=True)
    return imp(context, builder, sig, args)

def numpy_exp_scalar_input(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.exp, asfloat=True, scalar_input=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.exp, ty, types.Kind(types.Array))(numpy_exp_scalar_input))

def numpy_exp_scalar(context, builder, sig, args):
    imp = numpy_scalar_unary_ufunc(math.exp, asfloat=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.exp, ty)(numpy_exp_scalar))

# ------------------------------------------------------------------------------

@register
@implement(numpy.sin, types.Kind(types.Array), types.Kind(types.Array))
def numpy_sin(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.sin, asfloat=True)
    return imp(context, builder, sig, args)

def numpy_sin_scalar_input(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.sin, asfloat=True, scalar_input=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.sin, ty, types.Kind(types.Array))(numpy_sin_scalar_input))

def numpy_sin_scalar(context, builder, sig, args):
    imp = numpy_scalar_unary_ufunc(math.sin, asfloat=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.sin, ty)(numpy_sin_scalar))


@register
@implement(numpy.cos, types.Kind(types.Array), types.Kind(types.Array))
def numpy_cos(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.cos, asfloat=True)
    return imp(context, builder, sig, args)

def numpy_cos_scalar_input(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.cos, asfloat=True, scalar_input=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.cos, ty, types.Kind(types.Array))(numpy_cos_scalar_input))

def numpy_cos_scalar(context, builder, sig, args):
    imp = numpy_scalar_unary_ufunc(math.cos, asfloat=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.cos, ty)(numpy_cos_scalar))


@register
@implement(numpy.tan, types.Kind(types.Array), types.Kind(types.Array))
def numpy_tan(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.tan, asfloat=True)
    return imp(context, builder, sig, args)

def numpy_tan_scalar_input(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.tan, asfloat=True, scalar_input=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.tan, ty, types.Kind(types.Array))(numpy_tan_scalar_input))

def numpy_tan_scalar(context, builder, sig, args):
    imp = numpy_scalar_unary_ufunc(math.tan, asfloat=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.tan, ty)(numpy_tan_scalar))

# ------------------------------------------------------------------------------

@register
@implement(numpy.sinh, types.Kind(types.Array), types.Kind(types.Array))
def numpy_sinh(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.sinh, asfloat=True)
    return imp(context, builder, sig, args)

def numpy_sinh_scalar_input(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.sinh, asfloat=True, scalar_input=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.sinh, ty, types.Kind(types.Array))(numpy_sinh_scalar_input))

def numpy_sinh_scalar(context, builder, sig, args):
    imp = numpy_scalar_unary_ufunc(math.sinh, asfloat=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.sinh, ty)(numpy_sinh_scalar))


@register
@implement(numpy.cosh, types.Kind(types.Array), types.Kind(types.Array))
def numpy_cosh(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.cosh, asfloat=True)
    return imp(context, builder, sig, args)

def numpy_cosh_scalar_input(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.cosh, asfloat=True, scalar_input=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.cosh, ty, types.Kind(types.Array))(numpy_cosh_scalar_input))

def numpy_cosh_scalar(context, builder, sig, args):
    imp = numpy_scalar_unary_ufunc(math.cosh, asfloat=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.cosh, ty)(numpy_cosh_scalar))


@register
@implement(numpy.tanh, types.Kind(types.Array), types.Kind(types.Array))
def numpy_tanh(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.tanh, asfloat=True)
    return imp(context, builder, sig, args)

def numpy_tanh_scalar_input(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.tanh, asfloat=True, scalar_input=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.tanh, ty, types.Kind(types.Array))(numpy_tanh_scalar_input))

def numpy_tanh_scalar(context, builder, sig, args):
    imp = numpy_scalar_unary_ufunc(math.tanh, asfloat=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.tanh, ty)(numpy_tanh_scalar))

# ------------------------------------------------------------------------------

@register
@implement(numpy.arccos, types.Kind(types.Array), types.Kind(types.Array))
def numpy_arccos(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.acos, asfloat=True)
    return imp(context, builder, sig, args)

def numpy_arccos_scalar_input(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.acos, asfloat=True, scalar_input=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.arccos, ty, types.Kind(types.Array))(numpy_arccos_scalar_input))

def numpy_arccos_scalar(context, builder, sig, args):
    imp = numpy_scalar_unary_ufunc(math.acos, asfloat=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.arccos, ty)(numpy_arccos_scalar))

@register
@implement(numpy.arcsin, types.Kind(types.Array), types.Kind(types.Array))
def numpy_arcsin(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.asin, asfloat=True)
    return imp(context, builder, sig, args)

def numpy_arcsin_scalar_input(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.asin, asfloat=True, scalar_input=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.arcsin, ty, types.Kind(types.Array))(numpy_arcsin_scalar_input))

def numpy_arcsin_scalar(context, builder, sig, args):
    imp = numpy_scalar_unary_ufunc(math.asin, asfloat=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.arcsin, ty)(numpy_arcsin_scalar))


@register
@implement(numpy.arctan, types.Kind(types.Array), types.Kind(types.Array))
def numpy_arctan(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.atan, asfloat=True)
    return imp(context, builder, sig, args)

def numpy_arctan_scalar_input(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.atan, asfloat=True, scalar_input=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.arctan, ty, types.Kind(types.Array))(numpy_arctan_scalar_input))

def numpy_arctan_scalar(context, builder, sig, args):
    imp = numpy_scalar_unary_ufunc(math.atan, asfloat=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.arctan, ty)(numpy_arctan_scalar))

# ------------------------------------------------------------------------------

@register
@implement(numpy.arccosh, types.Kind(types.Array), types.Kind(types.Array))
def numpy_arccosh(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.acosh, asfloat=True)
    return imp(context, builder, sig, args)

def numpy_arccosh_scalar_input(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.acosh, asfloat=True, scalar_input=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.arccosh, ty, types.Kind(types.Array))(numpy_arccosh_scalar_input))

def numpy_arccosh_scalar(context, builder, sig, args):
    imp = numpy_scalar_unary_ufunc(math.acosh, asfloat=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.arccosh, ty)(numpy_arccosh_scalar))

@register
@implement(numpy.arcsinh, types.Kind(types.Array), types.Kind(types.Array))
def numpy_arcsinh(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.asinh, asfloat=True)
    return imp(context, builder, sig, args)

def numpy_arcsinh_scalar_input(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.asinh, asfloat=True, scalar_input=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.arcsinh, ty, types.Kind(types.Array))(numpy_arcsinh_scalar_input))

def numpy_arcsinh_scalar(context, builder, sig, args):
    imp = numpy_scalar_unary_ufunc(math.asinh, asfloat=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.arcsinh, ty)(numpy_arcsinh_scalar))


@register
@implement(numpy.arctanh, types.Kind(types.Array), types.Kind(types.Array))
def numpy_arctanh(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.atanh, asfloat=True)
    return imp(context, builder, sig, args)

def numpy_arctanh_scalar_input(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.atanh, asfloat=True, scalar_input=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.arctanh, ty, types.Kind(types.Array))(numpy_arctanh_scalar_input))

def numpy_arctanh_scalar(context, builder, sig, args):
    imp = numpy_scalar_unary_ufunc(math.atanh, asfloat=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.arctanh, ty)(numpy_arctanh_scalar))

# ------------------------------------------------------------------------------

@register
@implement(numpy.sqrt, types.Kind(types.Array), types.Kind(types.Array))
def numpy_sqrt(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.sqrt, asfloat=True)
    return imp(context, builder, sig, args)

def numpy_sqrt_scalar_input(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.sqrt, asfloat=True, scalar_input=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.sqrt, ty, types.Kind(types.Array))(numpy_sqrt_scalar_input))

def numpy_sqrt_scalar(context, builder, sig, args):
    imp = numpy_scalar_unary_ufunc(math.sqrt, asfloat=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.sqrt, ty)(numpy_sqrt_scalar))


@register
@implement(numpy.negative, types.Kind(types.Array), types.Kind(types.Array))
def numpy_negative(context, builder, sig, args):
    imp = numpy_unary_ufunc(types.neg_type)
    return imp(context, builder, sig, args)

def numpy_negative_scalar_input(context, builder, sig, args):
    imp = numpy_unary_ufunc(types.neg_type, scalar_input=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.negative, ty, types.Kind(types.Array))(numpy_negative_scalar_input))

def numpy_negative_scalar(context, builder, sig, args):
    imp = numpy_scalar_unary_ufunc(types.neg_type)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.negative, ty)(numpy_negative_scalar))


@register
@implement(numpy.floor, types.Kind(types.Array), types.Kind(types.Array))
def numpy_floor(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.floor, asfloat=True)
    return imp(context, builder, sig, args)

def numpy_floor_scalar_input(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.floor, scalar_input=True, asfloat=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.floor, ty, types.Kind(types.Array))(numpy_floor_scalar_input))

def numpy_floor_scalar(context, builder, sig, args):
    imp = numpy_scalar_unary_ufunc(math.floor, asfloat=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.floor, ty)(numpy_floor_scalar))


@register
@implement(numpy.ceil, types.Kind(types.Array), types.Kind(types.Array))
def numpy_ceil(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.ceil, asfloat=True)
    return imp(context, builder, sig, args)

def numpy_ceil_scalar_input(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.ceil, scalar_input=True, asfloat=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.ceil, ty, types.Kind(types.Array))(numpy_ceil_scalar_input))

def numpy_ceil_scalar(context, builder, sig, args):
    imp = numpy_scalar_unary_ufunc(math.ceil, asfloat=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.ceil, ty)(numpy_ceil_scalar))


@register
@implement(numpy.trunc, types.Kind(types.Array), types.Kind(types.Array))
def numpy_trunc(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.trunc, asfloat=True)
    return imp(context, builder, sig, args)

def numpy_trunc_scalar_input(context, builder, sig, args):
    imp = numpy_unary_ufunc(math.trunc, scalar_input=True, asfloat=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.trunc, ty, types.Kind(types.Array))(numpy_trunc_scalar_input))

def numpy_trunc_scalar(context, builder, sig, args):
    imp = numpy_scalar_unary_ufunc(math.trunc, asfloat=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.trunc, ty)(numpy_trunc_scalar))


@register
@implement(numpy.sign, types.Kind(types.Array), types.Kind(types.Array))
def numpy_sign(context, builder, sig, args):
    imp = numpy_unary_ufunc(types.sign_type)
    return imp(context, builder, sig, args)

def numpy_sign_scalar_input(context, builder, sig, args):
    imp = numpy_unary_ufunc(types.sign_type, scalar_input=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.sign, ty, types.Kind(types.Array))(numpy_sign_scalar_input))

def numpy_sign_scalar(context, builder, sig, args):
    imp = numpy_scalar_unary_ufunc(types.sign_type)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.sign, ty)(numpy_sign_scalar))


# TODO: handle mixed input types,
# and handle and one input op as scalar and other input op as array
def numpy_binary_ufunc(funckey, divbyzero=False, scalar_inputs=False,
                       asfloat=False):
    def impl(context, builder, sig, args):
        [tyinp1, tyinp2, tyout] = sig.args
        [inp1, inp2, out] = args
        if scalar_inputs:
            ndim = 1
        else:
            ndim = tyinp1.ndim

        # Temporary hack for __ftol2 llvm bug. Don't allow storing
        # float results in uint64 array on windows.
        if scalar_inputs and tyinp1 in types.real_domain and \
                tyout.dtype is types.uint64 and \
                sys.platform.startswith('win32'):
            raise TypeError('Cannot store result in uint64 array')
        if not scalar_inputs and tyinp1.dtype in types.real_domain and \
                tyout.dtype is types.uint64 and \
                sys.platform.startswith('win32'):
            raise TypeError('Cannot store result in uint64 array')

        if not scalar_inputs:
            i1ary = context.make_array(tyinp1)(context, builder, inp1)
            i2ary = context.make_array(tyinp2)(context, builder, inp2)
        oary = context.make_array(tyout)(context, builder, out)

        if asfloat and not divbyzero:
            sig = typing.signature(types.float64, types.float64, types.float64)
        else:
            if scalar_inputs:
                sig = typing.signature(tyout.dtype, tyinp1, tyinp2)
            else:
                sig = typing.signature(tyout.dtype, tyinp1.dtype, tyinp2.dtype)

        fnwork = context.get_function(funckey, sig)
        intpty = context.get_value_type(types.intp)

        # TODO handle differing shape by mimicking broadcasting
        loopshape = cgutils.unpack_tuple(builder, oary.shape, ndim)

        if scalar_inputs:
            xyo_shape = [cgutils.unpack_tuple(builder, ary.shape, ndim)
                         for ary in (oary,)]
            xyo_strides = [cgutils.unpack_tuple(builder, ary.strides, ndim)
                           for ary in (oary,)]
            xyo_data = [ary.data for ary in (oary,)]
            xyo_layout = [ty.layout for ty in (tyout,)]
        else:
            xyo_shape = [cgutils.unpack_tuple(builder, ary.shape, ndim)
                         for ary in (i1ary, i2ary, oary)]
            xyo_strides = [cgutils.unpack_tuple(builder, ary.strides, ndim)
                           for ary in (i1ary, i2ary, oary)]
            xyo_data = [ary.data for ary in (i1ary, i2ary, oary)]
            xyo_layout = [ty.layout for ty in (tyinp1, tyinp2, tyout)]

        with cgutils.loop_nest(builder, loopshape, intp=intpty) as indices:
            if scalar_inputs:
                [po] = [cgutils.get_item_pointer2(builder,
                                               data=data, shape=shape,
                                               strides=strides,
                                               layout=layout,
                                               inds=indices)
                                for data, shape, strides, layout
                                in zip(xyo_data, xyo_shape, xyo_strides,
                                       xyo_layout)]
            else:
                [px, py, po] = [cgutils.get_item_pointer2(builder,
                                                          data=data, shape=shape,
                                                          strides=strides,
                                                          layout=layout,
                                                          inds=indices)
                                for data, shape, strides, layout
                                in zip(xyo_data, xyo_shape, xyo_strides,
                                       xyo_layout)]

            if scalar_inputs:
                x = inp1
                y = inp2
            else:
                x = builder.load(px)
                y = builder.load(py)
            if divbyzero:
                # Handle division
                iszero = cgutils.is_scalar_zero(builder, y)
                with cgutils.ifelse(builder, iszero, expect=False) as (then,
                                                                       orelse):
                    with then:
                        # Divide by zero
                        if ((scalar_inputs and tyinp2 in types.real_domain) or
                                (not scalar_inputs and
                                    tyinp2.dtype in types.real_domain) or
                                not numpy_support.int_divbyzero_returns_zero):
                            # If y is float and is 0 also, return Nan; else
                            # return Inf
                            outltype = context.get_data_type(tyout.dtype)
                            shouldretnan = cgutils.is_scalar_zero(builder, x)
                            nan = Constant.real(outltype, float("nan"))
                            inf = Constant.real(outltype, float("inf"))
                            res = builder.select(shouldretnan, nan, inf)
                        elif (scalar_inputs and tyout in types.signed_domain and
                                not numpy_support.int_divbyzero_returns_zero):
                            res = Constant.int(context.get_data_type(tyout),
                                               0x1 << (y.type.width-1))
                        elif (not scalar_inputs and
                                tyout.dtype in types.signed_domain and
                                not numpy_support.int_divbyzero_returns_zero):
                            res = Constant.int(context.get_data_type(tyout.dtype),
                                               0x1 << (y.type.width-1))
                        else:
                            res = Constant.null(context.get_data_type(tyout.dtype))

                        assert res.type == po.type.pointee, \
                                        (str(res.type), str(po.type.pointee))
                        builder.store(res, po)
                    with orelse:
                        # Normal
                        tempres = fnwork(builder, (x, y))
                        if scalar_inputs and tyinp1 in types.real_domain:
                            res = context.cast(builder, tempres,
                                               tyinp1, tyout.dtype)
                        elif (not scalar_inputs and
                                tyinp1.dtype in types.real_domain):
                            res = context.cast(builder, tempres,
                                               tyinp1.dtype, tyout.dtype)
                        else:
                            res = context.cast(builder, tempres,
                                               types.float64, tyout.dtype)
                        assert res.type == po.type.pointee, \
                                        (str(res.type), str(po.type.pointee))
                        builder.store(res, po)
            else:
                # Handle non-division operations
                if asfloat:
                    if scalar_inputs:
                        d_x = context.cast(builder, x, tyinp1, types.float64)
                        d_y = context.cast(builder, y, tyinp2, types.float64)
                    else:
                        d_x = context.cast(builder, x, tyinp1.dtype,
                                           types.float64)
                        d_y = context.cast(builder, y, tyinp2.dtype,
                                           types.float64)
                    tempres = fnwork(builder, [d_x, d_y])
                    res = context.cast(builder, tempres,
                                       types.float64, tyout.dtype)
                elif scalar_inputs:
                    if tyinp1 != tyout.dtype:
                        tempres = fnwork(builder, [x, y])
                        res = context.cast(builder, tempres, tyinp1,
                                           tyout.dtype)
                    else:
                        res = fnwork(builder, (x, y))
                elif tyinp1.dtype != tyout.dtype:
                    tempres = fnwork(builder, [x, y])
                    res = context.cast(builder, tempres, tyinp1.dtype,
                                       tyout.dtype)
                else:
                    res = fnwork(builder, (x, y))
                assert res.type == po.type.pointee, (res.type,
                                                     po.type.pointee)
                builder.store(res, po)

        return out
    return impl


@register
@implement(numpy.add, types.Kind(types.Array), types.Kind(types.Array),
           types.Kind(types.Array))
def numpy_add(context, builder, sig, args):
    imp = numpy_binary_ufunc('+')
    return imp(context, builder, sig, args)

def numpy_add_scalar_inputs(context, builder, sig, args):
    imp = numpy_binary_ufunc('+', scalar_inputs=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.add, ty, ty,
             types.Kind(types.Array))(numpy_add_scalar_inputs))


@register
@implement(numpy.subtract, types.Kind(types.Array), types.Kind(types.Array),
           types.Kind(types.Array))
def numpy_subtract(context, builder, sig, args):
    imp = numpy_binary_ufunc('-')
    return imp(context, builder, sig, args)

def numpy_subtract_scalar_inputs(context, builder, sig, args):
    imp = numpy_binary_ufunc('-', scalar_inputs=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.subtract, ty, ty,
             types.Kind(types.Array))(numpy_subtract_scalar_inputs))


@register
@implement(numpy.multiply, types.Kind(types.Array), types.Kind(types.Array),
           types.Kind(types.Array))
def numpy_multiply(context, builder, sig, args):
    imp = numpy_binary_ufunc('*')
    return imp(context, builder, sig, args)

def numpy_multiply_scalar_inputs(context, builder, sig, args):
    imp = numpy_binary_ufunc('*', scalar_inputs=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.multiply, ty, ty,
             types.Kind(types.Array))(numpy_multiply_scalar_inputs))


'''@register
@implement(numpy.divide, types.Kind(types.Array), types.Kind(types.Array),
           types.Kind(types.Array))
def numpy_divide(context, builder, sig, args):
    dtype = sig.args[0].dtype
    odtype = sig.return_type.dtype
    isig = typing.signature(odtype, dtype, dtype)
    if dtype in types.signed_domain:
        if PYVERSION >= (3, 0):
            real_div_impl = context.get_function("/", isig)
            imp = numpy_binary_ufunc(real_div_impl, divbyzero=True)
        else:
            int_sdiv_impl = context.get_function("/?", isig)
            imp = numpy_binary_ufunc(int_sdiv_impl, divbyzero=True)
    elif dtype in types.unsigned_domain:
        int_udiv_impl = context.get_function("/?", isig)
        imp = numpy_binary_ufunc(int_udiv_impl, divbyzero=True)
    else:
        real_div_impl = context.get_function("/?", isig)
        imp = numpy_binary_ufunc(real_div_impl, divbyzero=True)

    return imp(context, builder, sig, args)'''


@register
@implement(numpy.divide, types.Kind(types.Array), types.Kind(types.Array),
           types.Kind(types.Array))
def numpy_divide(context, builder, sig, args):
    imp = numpy_binary_ufunc('/', asfloat=True, divbyzero=True)
    return imp(context, builder, sig, args)

def numpy_divide_scalar_inputs(context, builder, sig, args):
    imp = numpy_binary_ufunc('/', scalar_inputs=True, asfloat=True,
                             divbyzero=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.divide, ty, ty,
             types.Kind(types.Array))(numpy_divide_scalar_inputs))


@register
@implement(numpy.arctan2, types.Kind(types.Array), types.Kind(types.Array),
           types.Kind(types.Array))
def numpy_arctan2(context, builder, sig, args):
    imp = numpy_binary_ufunc(math.atan2, asfloat=True)
    return imp(context, builder, sig, args)

def numpy_arctan2_scalar_inputs(context, builder, sig, args):
    imp = numpy_binary_ufunc(math.atan2, asfloat=True, scalar_inputs=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.arctan2, ty, ty,
                       types.Kind(types.Array))(numpy_arctan2_scalar_inputs))

