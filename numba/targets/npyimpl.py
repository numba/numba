from __future__ import print_function, division, absolute_import

import numpy
import math
import sys
import itertools
from collections import namedtuple

from llvm.core import Constant, Type, ICMP_UGT


from .imputils import implement, Registry
from .. import typing, types, cgutils, numpy_support
from ..config import PYVERSION

registry = Registry()
register = registry.register


class npy:
    """This will be used as an index of the npy_* functions"""
    pass



def _decompose_type(ty, where='input operand'):
    """analyzes the type ty, returning a triplet containing:
    a boolean indicating if it is a scalar (at 0).
    the associated scalar type (at 1).
    the number of dimensions of the type (at 2).
    """
    if isinstance(ty, types.Array):
        return (False, ty.dtype, ty.ndim)
    elif ty in types.number_domain:
        return (True, ty, 1)
    else:
        raise TypeError('unknown type for {0}'.format(where))


def _default_promotion_for_type(ty):
    """returns the default type to be used when generating code
    associated to the type ty."""
    if ty in types.real_domain:
        promote_type = types.float64
    elif ty in types.signed_domain:
        promote_type = types.int64
    else:
        promote_type = types.uint64

    return promote_type


class _IndexingHelper(namedtuple('_IndexingHelper', ('array', 'indices'))):
    def update_indices(self, loop_indices, name):
        bld = self.array.builder
        intpty = self.array.context.get_value_type(types.intp)
        ONE = Constant.int(Type.int(intpty.width), 1)

        indices = loop_indices[len(loop_indices) - len(self.indices):]
        bb_index = [cgutils.append_basic_block(bld, '.inc_inp{0}_index{1}'.format(name, str(i))) for i in range(self.array.ndim)]
        bb_index.append(cgutils.append_basic_block(bld, 'end_inc{0}_index'.format(name)))
        bld.branch(bb_index[0])
        for i in range(self.array.ndim):
            with cgutils.goto_block(bld, bb_index[i]):
                cond = bld.icmp(ICMP_UGT, self.array.shape[i], ONE)
                with cgutils.ifthen(bld, cond):
                    bld.store(indices[i], self.indices[i])
                bld.branch(bb_index[i+1])
        bld.position_at_end(bb_index[-1])

    def dereference(self):
        bld = self.array.builder
        inds = [bld.load(index) for index in self.indices]
        px = cgutils.get_item_pointer2(bld,
                                       data=self.array.data,
                                       shape=self.array.shape,
                                       strides=self.array.strides,
                                       layout=self.array.layout,
                                       inds=inds)
        return  bld.load(px)


class _ArrayHelper(namedtuple('_ArrayHelper', ('context', 'builder', 'ary', 'shape', 'strides', 'data', 'layout', 'ndim'))):
    def create_iter_indices(self):
        intpty = self.context.get_value_type(types.intp)
        ZERO = Constant.int(Type.int(intpty.width), 0)

        indices = []
        for i in range(self.ndim):
            x = self.builder.alloca(Type.int(intpty.width))
            self.builder.store(ZERO, x)
            indices.append(x)
        return _IndexingHelper(self, indices)


def _prepare_array(ctxt, bld, inp, tyinp, ndim):
    """code that setups the array"""
    ary     = ctxt.make_array(tyinp)(ctxt, bld, inp)
    shape   = cgutils.unpack_tuple(bld, ary.shape, ndim)
    strides = cgutils.unpack_tuple(bld, ary.strides, ndim)
    return _ArrayHelper(ctxt, bld, ary, shape, strides, ary.data, tyinp.layout, ndim)


def unary_npy_math_extern(fn):
    setattr(npy, fn, fn)
    fn_sym = eval("npy."+fn)

    n = "numba.npymath." + fn
    def ref_impl(context, builder, sig, args):
        [val] = args
        mod = cgutils.get_module(builder)
        fnty = Type.function(Type.double(), [Type.double()])
        fn = mod.get_or_insert_function(fnty, name=n)
        return builder.call(fn, (val,))

    ty_dst = types.float64
    for ty_src in [types.int64, types.uint64, types.float64]:
        @register
        @implement(fn_sym, ty_src)
        def _impl(context, builder, sig, args):
            [val] = args
            cast_val = val if ty_dst == ty_src else context.cast(builder, val, ty_src, ty_dst)
            sig = typing.signature(ty_dst, ty_dst)
            return ref_impl(context, builder, sig, [cast_val])


def numpy_unary_ufunc(funckey, asfloat=False, scalar_input=False):
    def impl(context, builder, sig, args):
        [tyinp, tyout] = sig.args
        [inp, out] = args

        scalar_inp, scalar_tyinp, inp_ndim = _decompose_type(tyinp)

        promote_type = types.float64 if asfloat else _default_promotion_for_type(scalar_tyinp)
        result_type = promote_type

        # Temporary hack for __ftol2 llvm bug. Don't allow storing
        # float results in uint64 array on windows.
        if result_type in types.real_domain and \
                tyout.dtype is types.uint64 and \
                sys.platform.startswith('win32'):
            raise TypeError('Cannot store result in uint64 array')

        sig = typing.signature(result_type, promote_type)

        iary = None if scalar_inp else _prepare_array(context, builder, inp, tyinp, inp_ndim) 
        oary = _prepare_array(context, builder, out, tyout, tyout.ndim)

        fnwork = context.get_function(funckey, sig)
        intpty = context.get_value_type(types.intp)

        inp_indices = iary.create_iter_indices() if iary else None

        loopshape = oary.shape
        with cgutils.loop_nest(builder, loopshape, intp=intpty) as indices:
            if inp_indices is not None:
                inp_indices.update_indices(indices, '')

            x = inp_indices.dereference() if inp_indices else inp

            po = cgutils.get_item_pointer2(builder,
                                           data=oary.data,
                                           shape=oary.shape,
                                           strides=oary.strides,
                                           layout=oary.layout,
                                           inds=indices)

            d_x = context.cast(builder, x, scalar_tyinp, promote_type)
            tempres = fnwork(builder, [d_x])
            res = context.cast(builder, tempres, result_type, tyout.dtype)
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


def register_unary_ufunc(ufunc, operator, asfloat=False):

    def unary_ufunc(context, builder, sig, args):
        imp = numpy_unary_ufunc(operator, asfloat=asfloat)
        return imp(context, builder, sig, args)

    def unary_ufunc_scalar_input(context, builder, sig, args):
        imp = numpy_unary_ufunc(operator, scalar_input=True, asfloat=asfloat)
        return imp(context, builder, sig, args)

    def scalar_unary_ufunc(context, builder, sig, args):
        imp = numpy_scalar_unary_ufunc(operator, asfloat)
        return imp(context, builder, sig, args)

    register(implement(ufunc, types.Kind(types.Array),
        types.Kind(types.Array))(unary_ufunc))
    for ty in types.number_domain:
        register(implement(ufunc, ty,
            types.Kind(types.Array))(unary_ufunc_scalar_input))
    for ty in types.number_domain:
        register(implement(ufunc, ty)(scalar_unary_ufunc))



# _externs will be used to register ufuncs.
# each tuple contains the ufunc to be translated. That ufunc will be converted to
# an equivalent loop that calls the function in the npymath support module (registered
# as external function as "numba.npymath."+func
_externs = [
    (numpy.exp, "exp"),
    (numpy.exp2, "exp2"),
    (numpy.expm1, "expm1"),
    (numpy.log, "log"),
    (numpy.log2, "log2"),
    (numpy.log10, "log10"),
    (numpy.log1p, "log1p"),
    (numpy.deg2rad, "deg2rad"),
    (numpy.rad2deg, "rad2deg"),
    (numpy.sin, "sin"),
    (numpy.cos, "cos"),
    (numpy.tan, "tan"),
    (numpy.sinh, "sinh"),
    (numpy.cosh, "cosh"),
    (numpy.tanh, "tanh"),
    (numpy.arcsin, "asin"),
    (numpy.arccos, "acos"),
    (numpy.arctan, "atan"),
    (numpy.arcsinh, "asinh"),
    (numpy.arccosh, "acosh"),
    (numpy.arctanh, "atanh"),
    (numpy.sqrt, "sqrt"),
    (numpy.floor, "floor"),
    (numpy.ceil, "ceil"),
    (numpy.trunc, "trunc") ]


for x in _externs:
    unary_npy_math_extern(x[1])
    func = eval("npy." + x[1])
    register_unary_ufunc(x[0], func, asfloat = True)


register_unary_ufunc(numpy.absolute, types.abs_type)
register_unary_ufunc(numpy.sign, types.sign_type)
register_unary_ufunc(numpy.negative, types.neg_type)

def numpy_binary_ufunc(funckey, divbyzero=False, scalar_inputs=False,
                       asfloat=False, true_divide=False):
    def impl(context, builder, sig, args):
        [tyinp1, tyinp2, tyout] = sig.args
        [inp1, inp2, out] = args

        scalar_inp1, scalar_tyinp1, inp1_ndim = _decompose_type(tyinp1, where='first input operand')
        scalar_inp2, scalar_tyinp2, inp2_ndim = _decompose_type(tyinp2, where='second input operand')

        # based only on the first operand?
        promote_type = types.float64 if asfloat else _default_promotion_for_type(scalar_tyinp1)
        result_type = promote_type

        # Temporary hack for __ftol2 llvm bug. Don't allow storing
        # float results in uint64 array on windows.
        if result_type in types.real_domain and \
                tyout.dtype is types.uint64 and \
                sys.platform.startswith('win32'):
            raise TypeError('Cannot store result in uint64 array')

        sig = typing.signature(result_type, promote_type, promote_type)

        i1ary = None if scalar_inp1 else _prepare_array(context, builder, inp1, tyinp1, inp1_ndim) 
        i2ary = None if scalar_inp2 else _prepare_array(context, builder, inp2, tyinp2, inp2_ndim) 
        oary = _prepare_array(context, builder, out, tyout, tyout.ndim)

        fnwork = context.get_function(funckey, sig)
        intpty = context.get_value_type(types.intp)

        inp1_indices = i1ary.create_iter_indices() if i1ary else None
        inp2_indices = i2ary.create_iter_indices() if i2ary else None

        loopshape = oary.shape
        with cgutils.loop_nest(builder, loopshape, intp=intpty) as indices:
            if inp1_indices is not None:
                inp1_indices.update_indices(indices, '1')
            if inp2_indices is not None:
                inp2_indices.update_indices(indices, '2')

            x = inp1_indices.dereference() if inp1_indices else inp1
            y = inp2_indices.dereference() if inp2_indices else inp2

            po = cgutils.get_item_pointer2(builder,
                                           data=oary.data,
                                           shape=oary.shape,
                                           strides=oary.strides,
                                           layout=oary.layout,
                                           inds=indices)

            if divbyzero:
                # Handle division
                iszero = cgutils.is_scalar_zero(builder, y)
                with cgutils.ifelse(builder, iszero, expect=False) as (then,
                                                                       orelse):
                    with then:
                        # Divide by zero
                        if ((scalar_tyinp1 in types.real_domain or
                                scalar_tyinp2 in types.real_domain) or 
                                not numpy_support.int_divbyzero_returns_zero) or \
                                true_divide:
                            # If y is float and is 0 also, return Nan; else
                            # return Inf
                            outltype = context.get_data_type(result_type)
                            shouldretnan = cgutils.is_scalar_zero(builder, x)
                            nan = Constant.real(outltype, float("nan"))
                            inf = Constant.real(outltype, float("inf"))
                            tempres = builder.select(shouldretnan, nan, inf)
                            res = context.cast(builder, tempres, result_type,
                                               tyout.dtype)
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


def register_binary_ufunc(ufunc, operator, asfloat=False, divbyzero=False, true_divide=False):

    def binary_ufunc(context, builder, sig, args):
        imp = numpy_binary_ufunc(operator, asfloat=asfloat,
                                 divbyzero=divbyzero, true_divide=true_divide)
        return imp(context, builder, sig, args)

    def binary_ufunc_scalar_inputs(context, builder, sig, args):
        imp = numpy_binary_ufunc(operator, scalar_inputs=True, asfloat=asfloat,
                                 divbyzero=divbyzero, true_divide=true_divide)
        return imp(context, builder, sig, args)

    register(implement(ufunc, types.Kind(types.Array), types.Kind(types.Array),
        types.Kind(types.Array))(binary_ufunc))
    for ty in types.number_domain:
        register(implement(ufunc, ty, types.Kind(types.Array),
            types.Kind(types.Array))(binary_ufunc_scalar_inputs))
        register(implement(ufunc, types.Kind(types.Array), ty,
            types.Kind(types.Array))(binary_ufunc_scalar_inputs))
    for ty1, ty2 in itertools.product(types.number_domain, types.number_domain):
        register(implement(ufunc, ty1, ty2,
            types.Kind(types.Array))(binary_ufunc_scalar_inputs))

register_binary_ufunc(numpy.add, '+')
register_binary_ufunc(numpy.subtract, '-')
register_binary_ufunc(numpy.multiply, '*')
if not PYVERSION >= (3, 0):
    register_binary_ufunc(numpy.divide, '/', divbyzero=True, asfloat=True)
register_binary_ufunc(numpy.floor_divide, '//', divbyzero=True)
register_binary_ufunc(numpy.true_divide, '/', asfloat=True, divbyzero=True, true_divide=True)
register_binary_ufunc(numpy.arctan2, math.atan2, asfloat=True)
register_binary_ufunc(numpy.power, '**', asfloat=True)

