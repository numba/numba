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



class _ArrayHelper(namedtuple('_ArrayHelper', ('ary', 'shape', 'strides', 'data', 'layout'))):
    pass

def _prepare_array(ctxt, bld, inp, tyinp, ndim):
    """code that setups the array"""
    ary     = ctxt.make_array(tyinp)(ctxt, bld, inp)
    shape   = cgutils.unpack_tuple(bld, ary.shape, ndim)
    strides = cgutils.unpack_tuple(bld, ary.strides, ndim)
    return _ArrayHelper(ary, shape, strides, ary.data, tyinp.layout)


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

        out_ndim = tyout.ndim
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
        oary = _prepare_array(context, builder, out, tyout, out_ndim)

        fnwork = context.get_function(funckey, sig)
        intpty = context.get_value_type(types.intp)

        ZERO = Constant.int(Type.int(intpty.width), 0)
        ONE = Constant.int(Type.int(intpty.width), 1)

        inp_indices = None
        if not scalar_inp:
            inp_indices = []
            for i in range(inp_ndim):
                x = builder.alloca(Type.int(intpty.width))
                builder.store(ZERO, x)
                inp_indices.append(x)

        loopshape = oary.shape

        with cgutils.loop_nest(builder, loopshape, intp=intpty) as indices:

            # Increment input indices.
            # Since the output dimensions are already being incremented,
            # we'll use that to set the input indices. In order to
            # handle broadcasting, any input dimension of size 1 won't be
            # incremented.
            if iary is not None:
                bb_inc_inp_index = [cgutils.append_basic_block(builder,
                    '.inc_inp_index' + str(i)) for i in range(inp_ndim)]
                bb_end_inc_index = cgutils.append_basic_block(builder, '.end_inc_index')
                builder.branch(bb_inc_inp_index[0])
                for i in range(inp_ndim):
                    with cgutils.goto_block(builder, bb_inc_inp_index[i]):
                        # If the shape of this dimension is 1, then leave the
                        # index at 0 so that this dimension is broadcasted over
                        # the corresponding output dimension.
                        cond = builder.icmp(ICMP_UGT, iary.shape[i], ONE)
                        with cgutils.ifthen(builder, cond):
                            # If number of input dimensions is less than output
                            # dimensions, the input shape is right justified so
                            # that last dimension of input shape corresponds to
                            # last dimension of output shape. Therefore, index
                            # output dimension starting at offset of diff of
                            # input and output dimension count.
                            builder.store(indices[out_ndim-inp_ndim+i], inp_indices[i])
                        # We have to check if this is last dimension and add
                        # appropriate block terminator before beginning next
                        # loop.
                        if i + 1 == inp_ndim:
                            builder.branch(bb_end_inc_index)
                        else:
                            builder.branch(bb_inc_inp_index[i+1])
                builder.position_at_end(bb_end_inc_index)

                inds = [builder.load(index) for index in inp_indices]
                px = cgutils.get_item_pointer2(builder,
                                               data=iary.data,
                                               shape=iary.shape,
                                               strides=iary.strides,
                                               layout=iary.layout,
                                               inds=inds)
                x = builder.load(px)
            else:
                x = inp

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

        out_ndim = tyout.ndim

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
        oary = _prepare_array(context, builder, out, tyout, out_ndim)

        fnwork = context.get_function(funckey, sig)
        intpty = context.get_value_type(types.intp)

        ZERO = Constant.int(Type.int(intpty.width), 0)
        ONE = Constant.int(Type.int(intpty.width), 1)

        inp1_indices = None
        if not scalar_inp1:
            inp1_indices = []
            for i in range(inp1_ndim):
                x = builder.alloca(Type.int(intpty.width))
                builder.store(ZERO, x)
                inp1_indices.append(x)

        inp2_indices = None
        if not scalar_inp2:
            inp2_indices = []
            for i in range(inp2_ndim):
                x = builder.alloca(Type.int(intpty.width))
                builder.store(ZERO, x)
                inp2_indices.append(x)

        loopshape = oary.shape

        with cgutils.loop_nest(builder, loopshape, intp=intpty) as indices:

            # Increment input indices.
            # Since the output dimensions are already being incremented,
            # we'll use that to set the input indices. In order to
            # handle broadcasting, any input dimension of size 1 won't be
            # incremented.
            def build_increment_blocks(inp, inp_indices, inp_ndim, op_idx):
                bb_inc_inp_index = [cgutils.append_basic_block(builder,
                                                               '.inc_inp{0}_index{1}'.format(op_idx, str(i))) for i in range(inp_ndim)]
                bb_end_inc_index = cgutils.append_basic_block(builder,
                                                              '.end_inc{0}_index'.format(op_idx))

                builder.branch(bb_inc_inp_index[0])
                for i in range(inp_ndim):
                    with cgutils.goto_block(builder, bb_inc_inp_index[i]):
                        # If the shape of this dimension is 1, then leave the
                        # index at 0 so that this dimension is broadcasted over
                        # the corresponding input and output dimensions.
                        cond = builder.icmp(ICMP_UGT, inp.shape[i], ONE)
                        with cgutils.ifthen(builder, cond):
                            builder.store(indices[out_ndim-inp_ndim+i], inp_indices[i])
                        if i + 1 == inp_ndim:
                            builder.branch(bb_end_inc_index)
                        else:
                            builder.branch(bb_inc_inp_index[i+1])

                builder.position_at_end(bb_end_inc_index)

            if i1ary is not None:
                build_increment_blocks(i1ary, inp1_indices, inp1_ndim, '1')
            if i2ary is not None:
                build_increment_blocks(i2ary, inp2_indices, inp2_ndim, '2')

            if i1ary is None:
                x = inp1
            else:
                inds = [builder.load(index) for index in inp1_indices]
                px = cgutils.get_item_pointer2(builder,
                                               data=i1ary.data,
                                               shape=i1ary.shape,
                                               strides=i1ary.strides,
                                               layout=i1ary.layout,
                                               inds=inds)
                x = builder.load(px)

            if i2ary is None:
                y = inp2
            else:
                inds = [builder.load(index) for index in inp2_indices]
                py = cgutils.get_item_pointer2(builder,
                                               data=i2ary.data,
                                               shape=i2ary.shape,
                                               strides=i2ary.strides,
                                               layout=i2ary.layout,
                                               inds=inds)
                y = builder.load(py)

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

