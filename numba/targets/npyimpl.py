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

########################################################################

# In the way we generate code, ufuncs work with scalar as well as
# with array arguments. The following helper classes help dealing
# with scalar and array arguments in a regular way.
#
# In short, the classes provide a uniform interface. The interface
# handles the indexing of as many dimensions as the array may have.
# For scalars, all indexing is ignored and when the value is read,
# the scalar is returned. For arrays code for actual indexing is
# generated and reading performs the appropriate indirection.

class _ScalarIndexingHelper(object):
    def update_indices(self, loop_indices, name):
        pass

    def as_values(self):
        pass


class _ScalarHelper(namedtuple('_ScalarHelper', ('context', 'builder', 'val', 'base_type'))):
    def create_iter_indices(self):
        return _ScalarIndexingHelper()

    def load_effective_address(self, indices):
        raise LoweringError('Can not get effective address of a scalar')

    def load_data(self, indices):
        return self.val

    def store_data(self, indices, val):
        raise LoweringError('Can not store in a scalar')

class _ArrayIndexingHelper(namedtuple('_ArrayIndexingHelper', ('array', 'indices'))):
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

    def as_values(self):
        """The indexing helper is built using alloca for each value, so it actually contains pointers
        to the actual indices to load. Note that update_indices assumes the same. This method returns
        the indices as values"""
        bld=self.array.builder
        return [bld.load(index) for index in self.indices]


class _ArrayHelper(namedtuple('_ArrayHelper', ('context', 'builder', 'ary', 'shape', 'strides', 'data', 'layout', 'base_type', 'ndim'))):
    def create_iter_indices(self):
        intpty = self.context.get_value_type(types.intp)
        ZERO = Constant.int(Type.int(intpty.width), 0)

        indices = []
        for i in range(self.ndim):
            x = self.builder.alloca(Type.int(intpty.width))
            self.builder.store(ZERO, x)
            indices.append(x)
        return _ArrayIndexingHelper(self, indices)

    def load_effective_address(self, indices):
        return cgutils.get_item_pointer2(self.builder,
                                         data=self.data,
                                         shape=self.shape,
                                         strides=self.strides,
                                         layout=self.layout,
                                         inds=indices)

    def load_data(self, indices):
        return self.builder.load(self.load_effective_address(indices))

    def store_data(self, indices, value):
        assert self.context.get_data_type(self.base_type) == value.type
        self.builder.store(value, self.load_effective_address(indices))


def _prepare_argument(ctxt, bld, inp, tyinp, where='input operand'):
    """returns an instance of the appropriate Helper (either
    _ScalarHelper or _ArrayHelper) class to handle the argument.
    using the polimorphic interface of the Helper classes, scalar
    and array cases can be handled with the same code"""
    if isinstance(tyinp, types.Array):
        ary     = ctxt.make_array(tyinp)(ctxt, bld, inp)
        shape   = cgutils.unpack_tuple(bld, ary.shape, tyinp.ndim)
        strides = cgutils.unpack_tuple(bld, ary.strides, tyinp.ndim)
        return _ArrayHelper(ctxt, bld, ary, shape, strides, ary.data, tyinp.layout, tyinp.dtype, tyinp.ndim)
    elif tyinp in types.number_domain:
        return _ScalarHelper(ctxt, bld, inp, tyinp)
    else:
        raise TypeError('unknown type for {0}'.format(where))


def numpy_unary_ufunc(context, builder, sig, args, funckey, asfloat=False):
    [tyinp, tyout] = sig.args
    [inp, out] = args

    iary = _prepare_argument(context, builder, inp, tyinp)
    oary = _prepare_argument(context, builder, out, tyout)

    promote_type = types.float64 if asfloat else _default_promotion_for_type(iary.base_type)
    result_type = promote_type

    # Temporary hack for __ftol2 llvm bug. Don't allow storing
    # float results in uint64 array on windows.
    if result_type in types.real_domain and \
       tyout.dtype is types.uint64 and \
       sys.platform.startswith('win32'):
        raise TypeError('Cannot store result in uint64 array')


    sig    = typing.signature(result_type, promote_type)
    fnwork = context.get_function(funckey, sig)
    intpty = context.get_value_type(types.intp)

    inp_indices = iary.create_iter_indices()

    loopshape = oary.shape
    with cgutils.loop_nest(builder, loopshape, intp=intpty) as indices:
        inp_indices.update_indices(indices, '')
        x = iary.load_data(inp_indices.as_values())

        d_x = context.cast(builder, x, iary.base_type, promote_type)
        tempres = fnwork(builder, [d_x])
        res = context.cast(builder, tempres, result_type, tyout.dtype)
        oary.store_data(indices, res)

    return out



def numpy_scalar_unary_ufunc(context, builder, sig, args, funckey, asfloat=True):
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


def register_unary_ufunc(ufunc, operator, asfloat=False):

    def unary_ufunc(context, builder, sig, args):
        return numpy_unary_ufunc(context, builder, sig, args, operator, asfloat=asfloat)

    def scalar_unary_ufunc(context, builder, sig, args):
        return numpy_scalar_unary_ufunc(context, builder, sig, args, operator, asfloat)

    register(implement(ufunc, types.Kind(types.Array),
        types.Kind(types.Array))(unary_ufunc))
    for ty in types.number_domain:
        register(implement(ufunc, ty,
            types.Kind(types.Array))(unary_ufunc))
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
    (numpy.trunc, "trunc"),
]

_externs_2 = [
    (numpy.logaddexp, "logaddexp"),
    (numpy.logaddexp2, "logaddexp2"),
]

def npy_math_extern(fn, fnty):
    setattr(npy, fn, fn)
    fn_sym = eval("npy."+fn)
    fn_arity = len(fnty.args)

    n = "numba.npymath." + fn
    def ref_impl(context, builder, sig, args):
        mod = cgutils.get_module(builder)
        inner_fn = mod.get_or_insert_function(fnty, name=n)
        return builder.call(inner_fn, args)

    # This registers the function using different combinations of
    # input types that can be cast to the actual function type.
    #
    # Current limitation is that it only does so for homogeneous
    # source types. Note that it may be a better idea not providing
    # these specialization and let the ufunc generator functions
    # insert the appropriate castings before calling.
    #
    # TODO:
    # Either let the function only register the native version without
    # cast or provide the full range of specializations for functions
    # with arity > 1.
    ty_dst = types.float64
    for ty_src in [types.int64, types.uint64, types.float64]:
        @register
        @implement(fn_sym, *[ty_src]*fn_arity)
        def _impl(context, builder, sig, args):
            cast_vals = args if ty_dst == ty_src else [context.cast(builder, val, ty_src, ty_dst) for val in args]
            sig = typing.signature(*[ty_dst]*(len(cast_vals)+1))
            return ref_impl(context, builder, sig, cast_vals)


for sym, name in _externs:
    ty = Type.function(Type.double(), [Type.double()])
    npy_math_extern(name, ty)
    func = eval("npy." + name)
    register_unary_ufunc(sym, func, asfloat = True)

# radiams and degrees ufuncs are equivalent to deg2rad and rad2deg resp.
# register them.
register_unary_ufunc(numpy.degrees, numpy.rad2deg, asfloat=True)
register_unary_ufunc(numpy.radians, numpy.deg2rad, asfloat=True)

# the following ufuncs rely on functions that are not based on a function
# from npymath
register_unary_ufunc(numpy.absolute, types.abs_type)
register_unary_ufunc(numpy.sign, types.sign_type)
register_unary_ufunc(numpy.negative, types.neg_type)

def numpy_binary_ufunc(context, builder, sig, args, funckey, divbyzero=False,
                       asfloat=False, true_divide=False):
    [tyinp1, tyinp2, tyout] = sig.args
    [inp1, inp2, out] = args

    i1ary = _prepare_argument(context, builder, inp1, tyinp1)
    i2ary = _prepare_argument(context, builder, inp2, tyinp2)
    oary  = _prepare_argument(context, builder, out, tyout)

    # based only on the first operand?
    promote_type = types.float64 if asfloat else _default_promotion_for_type(i1ary.base_type)
    result_type = promote_type

    # Temporary hack for __ftol2 llvm bug. Don't allow storing
    # float results in uint64 array on windows.
    if result_type in types.real_domain and \
            tyout.dtype is types.uint64 and \
            sys.platform.startswith('win32'):
        raise TypeError('Cannot store result in uint64 array')

    sig = typing.signature(result_type, promote_type, promote_type)

    fnwork = context.get_function(funckey, sig)
    intpty = context.get_value_type(types.intp)

    inp1_indices = i1ary.create_iter_indices() if i1ary else None
    inp2_indices = i2ary.create_iter_indices() if i2ary else None

    loopshape = oary.shape
    with cgutils.loop_nest(builder, loopshape, intp=intpty) as indices:
        inp1_indices.update_indices(indices, '1')
        inp2_indices.update_indices(indices, '2')

        x = i1ary.load_data(inp1_indices.as_values())
        y = i2ary.load_data(inp2_indices.as_values())

        if divbyzero:
            # Handle division
            iszero = cgutils.is_scalar_zero(builder, y)
            with cgutils.ifelse(builder, iszero, expect=False) as (then,
                                                                   orelse):
                with then:
                    # Divide by zero
                    if ((i1ary.base_type in types.real_domain or
                            i2ary.base_type in types.real_domain) or
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

                    oary.store_data(indices, res)
                with orelse:
                    # Normal
                    d_x = context.cast(builder, x, i1ary.base_type, promote_type)
                    d_y = context.cast(builder, y, i2ary.base_type, promote_type)
                    tempres = fnwork(builder, [d_x, d_y])
                    res = context.cast(builder, tempres, result_type, tyout.dtype)
                    oary.store_data(indices, res)
        else:
            # Handle non-division operations
            d_x = context.cast(builder, x, i1ary.base_type, promote_type)
            d_y = context.cast(builder, y, i2ary.base_type, promote_type)
            tempres = fnwork(builder, [d_x, d_y])
            res = context.cast(builder, tempres, result_type, tyout.dtype)
            oary.store_data(indices, res)
    return out


def register_binary_ufunc(ufunc, operator, asfloat=False, divbyzero=False,
                          true_divide=False):

    def binary_ufunc(context, builder, sig, args):
        return numpy_binary_ufunc(context, builder, sig, args, operator,
                                  asfloat=asfloat, divbyzero=divbyzero, 
                                  true_divide=true_divide)

    register(implement(ufunc, types.Kind(types.Array), types.Kind(types.Array),
        types.Kind(types.Array))(binary_ufunc))
    for ty in types.number_domain:
        register(implement(ufunc, ty, types.Kind(types.Array),
            types.Kind(types.Array))(binary_ufunc))
        register(implement(ufunc, types.Kind(types.Array), ty,
            types.Kind(types.Array))(binary_ufunc))
    for ty1, ty2 in itertools.product(types.number_domain, types.number_domain):
        register(implement(ufunc, ty1, ty2,
            types.Kind(types.Array))(binary_ufunc))


register_binary_ufunc(numpy.add, '+')
register_binary_ufunc(numpy.subtract, '-')
register_binary_ufunc(numpy.multiply, '*')
if not PYVERSION >= (3, 0):
    register_binary_ufunc(numpy.divide, '/', divbyzero=True, asfloat=True)
register_binary_ufunc(numpy.floor_divide, '//', divbyzero=True)
register_binary_ufunc(numpy.true_divide, '/', asfloat=True, divbyzero=True, true_divide=True)
register_binary_ufunc(numpy.arctan2, math.atan2, asfloat=True)
register_binary_ufunc(numpy.power, '**', asfloat=True)


for sym, name in _externs_2:
    ty = Type.function(Type.double(), [Type.double(), Type.double()])
    npy_math_extern(name, ty)
    func = eval("npy." + name)
    register_binary_ufunc(sym, func, asfloat = True)
