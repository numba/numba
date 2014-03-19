from __future__ import print_function, division, absolute_import
import numpy
import math
import sys
from llvm.core import Constant, Type, ICMP_UGT
from numba import typing, types, cgutils
from numba.targets.imputils import implement, Registry
from numba import numpy_support
import itertools

registry = Registry()
register = registry.register


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
        sig = typing.signature(types.float64, types.float64)
        return f64impl(context, builder, sig, [fpval])

    @register
    @implement(fn_sym, types.uint64)
    def u64impl(context, builder, sig, args):
        [val] = args
        fpval = builder.uitofp(val, Type.double())
        sig = typing.signature(types.float64, types.float64)
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

_externs = [ "exp2", "expm1", "log", "log2", "log10", "log1p", "deg2rad", "rad2deg" ]
for x in _externs:
    unary_npy_math_extern(x)


def numpy_unary_ufunc(funckey, asfloat=False, scalar_input=False):
    def impl(context, builder, sig, args):
        [tyinp, tyout] = sig.args
        [inp, out] = args

        if isinstance(tyinp, types.Array):
            scalar_inp = False
            scalar_tyinp = tyinp.dtype
            inp_ndim = tyinp.ndim
        elif tyinp in types.number_domain:
            scalar_inp = True
            scalar_tyinp = tyinp
            inp_ndim = 1
        else:
            raise TypeError('unknown type for input operand')

        out_ndim = tyout.ndim

        if asfloat:
            promote_type = types.float64
        elif scalar_tyinp in types.real_domain:
            promote_type = types.float64
        elif scalar_tyinp in types.signed_domain:
            promote_type = types.int64
        else:
            promote_type = types.uint64

        result_type = promote_type

        sig = typing.signature(result_type, promote_type)

        if not scalar_inp:
            iary = context.make_array(tyinp)(context, builder, inp)
        oary = context.make_array(tyout)(context, builder, out)

        fnwork = context.get_function(funckey, sig)
        intpty = context.get_value_type(types.intp)

        if not scalar_inp:
            inp_shape = cgutils.unpack_tuple(builder, iary.shape, inp_ndim)
            inp_strides = cgutils.unpack_tuple(builder, iary.strides, inp_ndim)
            inp_data = iary.data
            inp_layout = tyinp.layout
        out_shape = cgutils.unpack_tuple(builder, oary.shape, out_ndim)
        out_strides = cgutils.unpack_tuple(builder, oary.strides, out_ndim)
        out_data = oary.data
        out_layout = tyout.layout

        ZERO = Constant.int(Type.int(intpty.width), 0)
        ONE = Constant.int(Type.int(intpty.width), 1)

        inp_indices = None
        if not scalar_inp:
            inp_indices = []
            for i in range(inp_ndim):
                x = builder.alloca(Type.int(intpty.width))
                builder.store(ZERO, x)
                inp_indices.append(x)

        loopshape = cgutils.unpack_tuple(builder, oary.shape, out_ndim)

        with cgutils.loop_nest(builder, loopshape, intp=intpty) as indices:

            # Increment input indices.
            # Since the output dimensions are already being incremented,
            # we'll use that to set the input indices. In order to
            # handle broadcasting, any input dimension of size 1 won't be
            # incremented.
            if not scalar_inp:
                bb_inc_inp_index = [cgutils.append_basic_block(builder,
                    '.inc_inp_index' + str(i)) for i in range(inp_ndim)]
                bb_end_inc_index = cgutils.append_basic_block(builder, '.end_inc_index')
                builder.branch(bb_inc_inp_index[0])
                for i in range(inp_ndim):
                    with cgutils.goto_block(builder, bb_inc_inp_index[i]):
                        # If the shape of this dimension is 1, then leave the
                        # index at 0 so that this dimension is broadcasted over
                        # the corresponding output dimension.
                        cond = builder.icmp(ICMP_UGT, inp_shape[i], ONE)
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
                                               data=inp_data,
                                               shape=inp_shape,
                                               strides=inp_strides,
                                               layout=inp_layout,
                                               inds=inds)
                x = builder.load(px)
            else:
                x = inp

            po = cgutils.get_item_pointer2(builder,
                                           data=out_data,
                                           shape=out_shape,
                                           strides=out_strides,
                                           layout=out_layout,
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

register_unary_ufunc(numpy.absolute, types.abs_type)
register_unary_ufunc(numpy.exp, math.exp, asfloat=True)
register_unary_ufunc(numpy.exp2, npy.exp2, asfloat=True)
register_unary_ufunc(numpy.expm1, npy.expm1, asfloat=True)
register_unary_ufunc(numpy.log, npy.log, asfloat=True)
register_unary_ufunc(numpy.log2, npy.log2, asfloat=True)
register_unary_ufunc(numpy.log10, npy.log10, asfloat=True)
register_unary_ufunc(numpy.log1p, npy.log1p, asfloat=True)
register_unary_ufunc(numpy.deg2rad, npy.deg2rad, asfloat=True)
register_unary_ufunc(numpy.rad2deg, npy.rad2deg, asfloat=True)
register_unary_ufunc(numpy.sin, math.sin, asfloat=True)
register_unary_ufunc(numpy.cos, math.cos, asfloat=True)
register_unary_ufunc(numpy.tan, math.tan, asfloat=True)
register_unary_ufunc(numpy.sinh, math.sinh, asfloat=True)
register_unary_ufunc(numpy.cosh, math.cosh, asfloat=True)
register_unary_ufunc(numpy.tanh, math.tanh, asfloat=True)
register_unary_ufunc(numpy.arcsin, math.asin, asfloat=True)
register_unary_ufunc(numpy.arccos, math.acos, asfloat=True)
register_unary_ufunc(numpy.arctan, math.atan, asfloat=True)
register_unary_ufunc(numpy.arcsinh, math.asinh, asfloat=True)
register_unary_ufunc(numpy.arccosh, math.acosh, asfloat=True)
register_unary_ufunc(numpy.arctanh, math.atanh, asfloat=True)
register_unary_ufunc(numpy.sqrt, math.sqrt, asfloat=True)
register_unary_ufunc(numpy.negative, types.neg_type)
register_unary_ufunc(numpy.floor, math.floor, asfloat=True)
register_unary_ufunc(numpy.ceil, math.ceil, asfloat=True)
register_unary_ufunc(numpy.trunc, math.trunc, asfloat=True)
register_unary_ufunc(numpy.sign, types.sign_type)


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
            raise TypeError('unknown type for first input operand')

        if isinstance(tyinp2, types.Array):
            scalar_inp2 = False
            scalar_tyinp2 = tyinp2.dtype
            inp2_ndim = tyinp2.ndim
        elif tyinp2 in types.number_domain:
            scalar_inp2 = True
            scalar_tyinp2 = tyinp2
            inp2_ndim = 1
        else:
            raise TypeError('unknown type for second input operand')

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

        result_type = promote_type

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

        loopshape = cgutils.unpack_tuple(builder, oary.shape, out_ndim)

        with cgutils.loop_nest(builder, loopshape, intp=intpty) as indices:

            # Increment input indices.
            # Since the output dimensions are already being incremented,
            # we'll use that to set the input indices. In order to
            # handle broadcasting, any input dimension of size 1 won't be
            # incremented.
            def build_increment_blocks(inp_indices, inp_shape, inp_ndim, inp_num):
                bb_inc_inp_index = [cgutils.append_basic_block(builder,
                    '.inc_inp{0}_index{1}'.format(inp_num, str(i))) for i in range(inp_ndim)]
                bb_end_inc_index = cgutils.append_basic_block(builder,
                                       '.end_inc{0}_index'.format(inp_num))

                builder.branch(bb_inc_inp_index[0])
                for i in range(inp_ndim):
                    with cgutils.goto_block(builder, bb_inc_inp_index[i]):
                        # If the shape of this dimension is 1, then leave the
                        # index at 0 so that this dimension is broadcasted over
                        # the corresponding input and output dimensions.
                        cond = builder.icmp(ICMP_UGT, inp_shape[i], ONE)
                        with cgutils.ifthen(builder, cond):
                            builder.store(indices[out_ndim-inp_ndim+i], inp_indices[i])
                        if i + 1 == inp_ndim:
                            builder.branch(bb_end_inc_index)
                        else:
                            builder.branch(bb_inc_inp_index[i+1])

                builder.position_at_end(bb_end_inc_index)

            if not scalar_inp1:
                build_increment_blocks(inp1_indices, inp1_shape, inp1_ndim, '1')
            if not scalar_inp2:
                build_increment_blocks(inp2_indices, inp2_shape, inp2_ndim, '2')

            if scalar_inp1:
                x = inp1
            else:
                inds = [builder.load(index) for index in inp1_indices]
                px = cgutils.get_item_pointer2(builder,
                                               data=inp1_data,
                                               shape=inp1_shape,
                                               strides=inp1_strides,
                                               layout=inp1_layout,
                                               inds=inds)
                x = builder.load(px)

            if scalar_inp2:
                y = inp2
            else:
                inds = [builder.load(index) for index in inp2_indices]
                py = cgutils.get_item_pointer2(builder,
                                               data=inp2_data,
                                               shape=inp2_shape,
                                               strides=inp2_strides,
                                               layout=inp2_layout,
                                               inds=inds)
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


def register_binary_ufunc(ufunc, operator, asfloat=False, divbyzero=False):

    def binary_ufunc(context, builder, sig, args):
        imp = numpy_binary_ufunc(operator, asfloat=asfloat, divbyzero=divbyzero)
        return imp(context, builder, sig, args)

    def binary_ufunc_scalar_inputs(context, builder, sig, args):
        imp = numpy_binary_ufunc(operator, scalar_inputs=True, asfloat=asfloat,
                                 divbyzero=divbyzero)
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
register_binary_ufunc(numpy.divide, '/', asfloat=True, divbyzero=True)
register_binary_ufunc(numpy.arctan2, math.atan2, asfloat=True)

