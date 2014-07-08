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
    (numpy.arctan2, "atan2"),
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
            raise LoweringError("codepath no longer supported")
    return out


def numpy_ufunc_kernel(context, builder, sig, args, kernel_class):
    arguments = [_prepare_argument(context, builder, arg, tyarg)
                 for arg, tyarg in zip(args, sig.args)]
    inputs = arguments[0:-1]
    output = arguments[-1]
    outer_sig = [a.base_type for a in arguments]
    #signature expects return type first, while we have it last:
    outer_sig = outer_sig[-1:] + outer_sig[:-1]
    outer_sig = typing.signature(*outer_sig)
    kernel = kernel_class(context, builder, outer_sig)
    intpty = context.get_value_type(types.intp)

    indices = [inp.create_iter_indices() for inp in inputs]

    loopshape = output.shape
    with cgutils.loop_nest(builder, loopshape, intp=intpty) as loop_indices:
        vals_in = []
        for i, (index, arg) in enumerate(zip(indices, inputs)):
            index.update_indices(loop_indices, i)
            vals_in.append(arg.load_data(index.as_values()))

        val_out = kernel.generate(*vals_in)
        output.store_data(loop_indices, val_out)
    return args[-1]


# Kernels are the code to be executed inside the multidimensional loop.
class _Kernel(object):
    pass


def _function_with_cast(op, inner_sig):
    """a kernel implemented by a function that only exists in one signature
    op is the operation (function
    inner_sig is the signature of op. Operands will be cast to that signature
    """
    class _KernelImpl(_Kernel):
        def __init__(self, context, builder, outer_sig):
            """
            op is the operation
            outer_sig is the outer type signature (the signature of the ufunc)
            inner_sig is the inner type signature (the signature of the operation itself)
            """
            self.context = context
            self.builder = builder
            self.fnwork = context.get_function(op, inner_sig)
            self.inner_sig = inner_sig
            self.outer_sig = outer_sig

        def generate(self, *args):
            #convert args from the ufunc types to the one of the kernel operation
            cast_args = [self.context.cast(self.builder, val, inty, outy)
                         for val, inty, outy in zip(args, self.outer_sig.args,
                                                    self.inner_sig.args)]
            #perform the operation
            res = self.fnwork(self.builder, cast_args)
            #return the result converted to the type of the ufunc operation
            return self.context.cast(self.builder, res, self.inner_sig.return_type,
                                     self.outer_sig.return_type)

    return _KernelImpl


def _homogeneous_function(op):
    """A kernel using an homogeneous inner signature, based on the type of
    the first argument. All arguments will be cast to that type and the return
    type of the operation is assumed to have that type before casting"""
    class _KernelImpl(_Kernel):
        def __init__(self, context, builder, outer_sig):
            self.context = context
            self.builder = builder
            self.promote_type = _default_promotion_for_type(outer_sig.args[0])
            self.outer_sig = outer_sig

        def generate(self, *args):
            inner_sig = typing.signature(*([self.promote_type]*(len(self.outer_sig.args)+1)))
            fn = self.context.get_function(op, inner_sig)
            cast_args = [self.context.cast(self.builder, val, inty, outty)
                         for val, inty, outty in zip(args, self.outer_sig.args, inner_sig.args)]
            res = fn(self.builder, cast_args)
            return self.context.cast(self.builder, res, self.promote_type,
                                     self.outer_sig.return_type)

    return _KernelImpl


def _true_division():
    """A kernel for division. It supports three kinds of division:
    'true': true division as in python3
    'floor': regular floor division
    'plain':
    """
    inner_sig = typing.signature(types.float64, types.float64, types.float64)
    class _KernelImpl(_Kernel):
        def __init__(self, context, builder, outer_sig):
            self.context = context
            self.builder = builder
            self.outer_sig = outer_sig
            self.fnwork = context.get_function('/', inner_sig)

        def generate(self,*args):
            assert len(args) == 2 # numerator and denominator
            num, den = args
            iszero = cgutils.is_scalar_zero(self.builder, den)
            with cgutils.ifelse(self.builder, iszero, expect=False) as (then, orelse):
                outltype = self.context.get_data_type(types.float64)
                with then:
                    shouldretnan = cgutils.is_scalar_zero(self.builder, num)
                    nan = Constant.real(outltype, float("nan"))
                    inf = Constant.real(outltype, float("inf"))
                    res_then = self.builder.select(shouldretnan, nan, inf)
                    bb_then = self.builder.basic_block
                with orelse:
                    cast_args = [self.context.cast(self.builder, val, inty, types.float64)
                                 for val, inty in zip(args, self.outer_sig.args)]
                    res_else = self.fnwork(self.builder, cast_args)
                    bb_else = self.builder.basic_block

            res = self.builder.phi(outltype)
            res.add_incoming(res_then, bb_then)
            res.add_incoming(res_else, bb_else)

            return self.context.cast(self.builder, res, types.float64,
                                     self.outer_sig.return_type)

    return _KernelImpl

################################################################################
# Helper functions that register the ufuncs

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


def register_unary_ufunc_kernel(ufunc, kernel):
    def unary_ufunc(context, builder, sig, args):
        return numpy_ufunc_kernel(context, builder, sig, args, kernel)

    register(implement(ufunc, types.Kind(types.Array),
        types.Kind(types.Array))(unary_ufunc))
    for ty in types.number_domain:
        register(implement(ufunc, ty,
            types.Kind(types.Array))(unary_ufunc))
    for ty in types.number_domain:
        register(implement(ufunc, ty)(unary_ufunc))


def register_binary_ufunc_kernel(ufunc, kernel):
    def binary_ufunc(context, builder, sig, args):
        return numpy_ufunc_kernel(context, builder, sig, args, kernel)

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


################################################################################
# Actual registering of supported ufuncs

_float_unary_function_type = Type.function(Type.double(), [Type.double()])
_float_binary_function_type = Type.function(Type.double(), [Type.double(), Type.double()])
_float_unary_sig = typing.signature(types.float64, types.float64)
_float_binary_sig = typing.signature(types.float64, types.float64, types.float64)


for sym, name in _externs:
    npy_math_extern(name, _float_unary_function_type)
    register_unary_ufunc_kernel(sym, _function_with_cast(getattr(npy, name), _float_unary_sig))

# radians and degrees ufuncs are equivalent to deg2rad and rad2deg resp.
# register them.
register_unary_ufunc_kernel(numpy.degrees, _function_with_cast(npy.rad2deg, _float_unary_sig))
register_unary_ufunc_kernel(numpy.radians, _function_with_cast(npy.deg2rad, _float_unary_sig))

# the following ufuncs rely on functions that are not based on a function
# from npymath
register_unary_ufunc_kernel(numpy.absolute, _homogeneous_function(types.abs_type))
register_unary_ufunc_kernel(numpy.sign, _homogeneous_function(types.sign_type))
register_unary_ufunc_kernel(numpy.negative, _homogeneous_function(types.neg_type))

# for these we mostly rely on code generation for python operators.
register_binary_ufunc_kernel(numpy.add, _homogeneous_function('+'))
register_binary_ufunc_kernel(numpy.subtract, _homogeneous_function('-'))
register_binary_ufunc_kernel(numpy.multiply, _homogeneous_function('*'))
if not PYVERSION >= (3, 0):
    register_binary_ufunc(numpy.divide, '/', divbyzero=True, asfloat=True)
register_binary_ufunc(numpy.floor_divide, '//', divbyzero=True)
register_binary_ufunc_kernel(numpy.true_divide, _true_division())
register_binary_ufunc_kernel(numpy.power, _function_with_cast('**', _float_binary_sig))

for sym, name in _externs_2:
    npy_math_extern(name, _float_binary_function_type)
    register_binary_ufunc_kernel(sym, _function_with_cast(getattr(npy, name), _float_binary_sig))

del _float_binary_function_type, _float_binary_sig
del _float_unary_function_type, _float_unary_sig
