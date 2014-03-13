from __future__ import print_function, division, absolute_import
import numpy
import math
import sys
from llvm.core import Constant, Type
from numba.utils import PYVERSION
from numba import typing, types, cgutils
from numba.targets.imputils import implement
from numba import numpy_support
import itertools

functions = []


def register(f):
    functions.append(f)
    return f


class npy:
    """This will be used as an index of the npy_* functions"""
    pass

def unary_npy_math_extern(fn):
    setattr(npy, fn, fn)
    fn_sym = eval("npy."+fn)
    @register
    @implement(fn_sym, types.int64)
    def s64impl(context, builder, sig, args):
        [val] = args
        fpval = builder.sitofp(val, Type.double())
        sig = signature(types.float64, types.float64)
        return f64impl(context, builder, sig, [fpval])

    @register
    @implement(fn_sym, types.uint64)
    def u64impl(context, builder, sig, args):
        [val] = args
        fpval = builder.uitofp(val, Type.double())
        sig = signature(types.float64, types.float64)
        return f64impl(context, builder, sig, [fpval])

    n = "numba.numpy.math." + fn
    @register
    @implement(fn_sym, types.float64)
    def f64impl(context, builder, sig, args):
        [val] = args
        mod = cgutils.get_module(builder)
        fnty = Type.function(Type.double(), [Type.double()])
        fn = mod.get_or_insert_function(fnty, name=n)
        return builder.call(fn, (val,))


unary_npy_math_extern("exp2")
unary_npy_math_extern("log")

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


@register
@implement(numpy.exp2, types.Kind(types.Array), types.Kind(types.Array))
def numpy_exp2(context, builder, sig, args):
    imp = numpy_unary_ufunc(npy.exp2, asfloat=True)
    return imp(context, builder, sig, args)

def numpy_exp2_scalar_input(context, builder, sig, args):
    imp = numpy_unary_ufunc(npy.exp2, asfloat=True, scalar_input=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.exp2, ty,
                       types.Kind(types.Array)
                       )(numpy_exp2_scalar_input))


def numpy_exp2_scalar(context, builder, sig, args):
    imp = numpy_scalar_unary_ufunc(npy.exp2, asfloat=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.exp2, ty)(numpy_exp_scalar))


@register
@implement(numpy.log, types.Kind(types.Array), types.Kind(types.Array))
def numpy_log(context, builder, sig, args):
    imp = numpy_unary_ufunc(npy.log, asfloat=True)
    return imp(context, builder, sig, args)

def numpy_log_scalar_input(context, builder, sig, args):
    imp = numpy_unary_ufunc(npy.log, asfloat=True, scalar_input=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.log, ty,
                       types.Kind(types.Array)
                       )(numpy_log_scalar_input))


def numpy_log_scalar(context, builder, sig, args):
    imp = numpy_scalar_unary_ufunc(npy.log, asfloat=True)
    return imp(context, builder, sig, args)

for ty in types.number_domain:
    register(implement(numpy.log, ty)(numpy_exp_scalar))



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

        if isinstance(tyinp1, types.Array):
            scalar_inp1 = False
            scalar_tyinp1 = tyinp1.dtype
            inp1_ndim = tyinp1.ndim
        elif tyinp1 in types.number_domain:
            scalar_inp1 = True
            scalar_tyinp1 = tyinp1
            inp1_ndim = 1
        else:
            raise TypeError('unknown type for first operand')

        if isinstance(tyinp2, types.Array):
            scalar_inp2 = False
            scalar_tyinp2 = tyinp2.dtype
            inp2_ndim = tyinp2.ndim
        elif tyinp2 in types.number_domain:
            scalar_inp2 = True
            scalar_tyinp2 = tyinp2
            inp2_ndim = 1
        else:
            raise TypeError('unknown type for second operand')

        out_ndim = tyout.ndim

        if asfloat:
            promote_type = types.float64
        elif scalar_tyinp1 in types.real_domain or \
                scalar_tyinp2 in types.real_domain:
            promote_type = types.float64
        elif scalar_tyinp1 in types.signed_domain or \
                scalar_tyinp2 in types.signed_domain:
            promote_type = types.int64
        else:
            promote_type = types.uint64

        if promote_type == types.float64:
            result_type = types.float64
        elif promote_type == types.int64:
            result_type = types.int64
        else:
            result_type = types.uint64

        # Temporary hack for __ftol2 llvm bug. Don't allow storing
        # float results in uint64 array on windows.
        if result_type in types.real_domain and \
                tyout.dtype is types.uint64 and \
                sys.platform.startswith('win32'):
            raise TypeError('Cannot store result in uint64 array')
        
        sig = typing.signature(result_type, promote_type, promote_type)

        if not scalar_inp1:
            i1ary = context.make_array(tyinp1)(context, builder, inp1)
        if not scalar_inp2:
            i2ary = context.make_array(tyinp2)(context, builder, inp2)
        oary = context.make_array(tyout)(context, builder, out)

        fnwork = context.get_function(funckey, sig)
        intpty = context.get_value_type(types.intp)

        # TODO handle differing shape by mimicking broadcasting
        loopshape = cgutils.unpack_tuple(builder, oary.shape, out_ndim)

        if not scalar_inp1:
            inp1_shape = cgutils.unpack_tuple(builder, i1ary.shape, inp1_ndim)
            inp1_strides = cgutils.unpack_tuple(builder, i1ary.strides, inp1_ndim)
            inp1_data = i1ary.data
            inp1_layout = tyinp1.layout
        if not scalar_inp2:
            inp2_shape = cgutils.unpack_tuple(builder, i2ary.shape, inp2_ndim)
            inp2_strides = cgutils.unpack_tuple(builder, i2ary.strides, inp2_ndim)
            inp2_data = i2ary.data
            inp2_layout = tyinp2.layout
        out_shape = cgutils.unpack_tuple(builder, oary.shape, out_ndim)
        out_strides = cgutils.unpack_tuple(builder, oary.strides, out_ndim)
        out_data = oary.data
        out_layout = tyout.layout

        with cgutils.loop_nest(builder, loopshape, intp=intpty) as indices:
            
            if scalar_inp1:
                x = inp1
            else:
                px = cgutils.get_item_pointer2(builder,
                                               data=inp1_data,
                                               shape=inp1_shape,
                                               strides=inp1_strides,
                                               layout=inp1_layout,
                                               inds=indices)
                x = builder.load(px)

            if scalar_inp2:
                y = inp2
            else:
                py = cgutils.get_item_pointer2(builder,
                                               data=inp2_data,
                                               shape=inp2_shape,
                                               strides=inp2_strides,
                                               layout=inp2_layout,
                                               inds=indices)
                y = builder.load(py)

            po = cgutils.get_item_pointer2(builder,
                                           data=out_data,
                                           shape=out_shape,
                                           strides=out_strides,
                                           layout=out_layout,
                                           inds=indices)

            if divbyzero:
                # Handle division
                iszero = cgutils.is_scalar_zero(builder, y)
                with cgutils.ifelse(builder, iszero, expect=False) as (then,
                                                                       orelse):
                    with then:
                        # Divide by zero
                        if (scalar_tyinp1 in types.real_domain or
                                scalar_tyinp2 in types.real_domain) or \
                                not numpy_support.int_divbyzero_returns_zero:
                            # If y is float and is 0 also, return Nan; else
                            # return Inf
                            outltype = context.get_data_type(result_type)
                            shouldretnan = cgutils.is_scalar_zero(builder, x)
                            nan = Constant.real(outltype, float("nan"))
                            inf = Constant.real(outltype, float("inf"))
                            tempres = builder.select(shouldretnan, nan, inf)
                            res = context.cast(builder, tempres, result_type, tyout.dtype)
                        elif tyout.dtype in types.signed_domain and \
                                not numpy_support.int_divbyzero_returns_zero:
                            res = Constant.int(context.get_data_type(tyout.dtype),
                                               0x1 << (y.type.width-1))
                        else:
                            res = Constant.null(context.get_data_type(tyout.dtype))

                        assert res.type == po.type.pointee, \
                                        (str(res.type), str(po.type.pointee))
                        builder.store(res, po)
                    with orelse:
                        # Normal
                        d_x = context.cast(builder, x, scalar_tyinp1, promote_type)
                        d_y = context.cast(builder, y, scalar_tyinp2, promote_type)
                        tempres = fnwork(builder, [d_x, d_y])
                        res = context.cast(builder, tempres, result_type, tyout.dtype)

                        assert res.type == po.type.pointee, (res.type,
                                                             po.type.pointee)
                        builder.store(res, po)
            else:
                # Handle non-division operations
                d_x = context.cast(builder, x, scalar_tyinp1, promote_type)
                d_y = context.cast(builder, y, scalar_tyinp2, promote_type)
                tempres = fnwork(builder, [d_x, d_y])
                res = context.cast(builder, tempres, result_type, tyout.dtype)

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
    register(implement(numpy.add, ty, types.Kind(types.Array),
             types.Kind(types.Array))(numpy_add_scalar_inputs))
    register(implement(numpy.add, types.Kind(types.Array), ty,
             types.Kind(types.Array))(numpy_add_scalar_inputs))

for ty1, ty2 in itertools.product(types.number_domain, types.number_domain):
    register(implement(numpy.add, ty1, ty2,
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

for ty1, ty2 in itertools.product(types.number_domain, types.number_domain):
    register(implement(numpy.subtract, ty1, ty2,
             types.Kind(types.Array))(numpy_subtract_scalar_inputs))

for ty in types.number_domain:
    register(implement(numpy.subtract, ty, types.Kind(types.Array),
             types.Kind(types.Array))(numpy_subtract_scalar_inputs))
    register(implement(numpy.subtract, types.Kind(types.Array), ty,
             types.Kind(types.Array))(numpy_subtract_scalar_inputs))

for ty1, ty2 in itertools.product(types.number_domain, types.number_domain):
    register(implement(numpy.subtract, ty1, ty2,
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

for ty1, ty2 in itertools.product(types.number_domain, types.number_domain):
    register(implement(numpy.multiply, ty1, ty2,
             types.Kind(types.Array))(numpy_multiply_scalar_inputs))

for ty in types.number_domain:
    register(implement(numpy.multiply, ty, types.Kind(types.Array),
             types.Kind(types.Array))(numpy_multiply_scalar_inputs))
    register(implement(numpy.multiply, types.Kind(types.Array), ty,
             types.Kind(types.Array))(numpy_multiply_scalar_inputs))

for ty1, ty2 in itertools.product(types.number_domain, types.number_domain):
    register(implement(numpy.multiply, ty1, ty2,
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
    register(implement(numpy.divide, ty, types.Kind(types.Array),
             types.Kind(types.Array))(numpy_divide_scalar_inputs))
    register(implement(numpy.divide, types.Kind(types.Array), ty,
             types.Kind(types.Array))(numpy_divide_scalar_inputs))

for ty1, ty2 in itertools.product(types.number_domain, types.number_domain):
    register(implement(numpy.divide, ty1, ty2,
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

