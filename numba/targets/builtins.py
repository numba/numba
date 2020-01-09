from __future__ import print_function, absolute_import, division

import math
from functools import reduce

import numpy as np
import operator

from llvmlite import ir
from llvmlite.llvmpy.core import Type, Constant
import llvmlite.llvmpy.core as lc

from .imputils import (lower_builtin, lower_getattr, lower_getattr_generic,
                       lower_cast, lower_constant, iternext_impl,
                       call_getiter, call_iternext,
                       impl_ret_borrowed, impl_ret_untracked,
                       numba_typeref_ctor)
from .. import typing, types, cgutils, utils
from ..extending import overload, intrinsic
from numba.typeconv import Conversion
from numba.errors import TypingError


@overload(operator.truth)
def ol_truth(val):
    if isinstance(val, types.Boolean):
        def impl(val):
            return val
        return impl


@lower_builtin(operator.is_not, types.Any, types.Any)
def generic_is_not(context, builder, sig, args):
    """
    Implement `x is not y` as `not (x is y)`.
    """
    is_impl = context.get_function(operator.is_, sig)
    return builder.not_(is_impl(builder, args))


@lower_builtin(operator.is_, types.Any, types.Any)
def generic_is(context, builder, sig, args):
    """
    Default implementation for `x is y`
    """
    lhs_type, rhs_type = sig.args
    # the lhs and rhs have the same type
    if lhs_type == rhs_type:
            # mutable types
            if lhs_type.mutable:
                raise NotImplementedError('no default `is` implementation')
            # immutable types
            else:
                # fallbacks to `==`
                try:
                    eq_impl = context.get_function(operator.eq, sig)
                except NotImplementedError:
                    # no `==` implemented for this type
                    return cgutils.false_bit
                else:
                    return eq_impl(builder, args)
    else:
        return cgutils.false_bit


@lower_builtin(operator.eq, types.Literal, types.Literal)
@lower_builtin(operator.eq, types.IntegerLiteral, types.IntegerLiteral)
def const_eq_impl(context, builder, sig, args):
    arg1, arg2 = sig.args
    val = 0
    if arg1.literal_value == arg2.literal_value:
        val = 1
    res = ir.Constant(ir.IntType(1), val)
    return impl_ret_untracked(context, builder, sig.return_type, res)


@lower_builtin(operator.ne, types.StringLiteral, types.StringLiteral)
def const_ne_impl(context, builder, sig, args):
    arg1, arg2 = sig.args
    val = 0
    if arg1.literal_value != arg2.literal_value:
        val = 1
    res = ir.Constant(ir.IntType(1), val)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def gen_non_eq(val):
    def none_equality(a, b):
        a_none = isinstance(a, types.NoneType)
        b_none = isinstance(b, types.NoneType)
        if a_none and b_none:
            def impl(a, b):
                return val
            return impl
        elif a_none ^ b_none:
            def impl(a, b):
                return not val
            return impl
    return none_equality

overload(operator.eq)(gen_non_eq(True))
overload(operator.ne)(gen_non_eq(False))

#-------------------------------------------------------------------------------

@lower_getattr_generic(types.DeferredType)
def deferred_getattr(context, builder, typ, value, attr):
    """
    Deferred.__getattr__ => redirect to the actual type.
    """
    inner_type = typ.get()
    val = context.cast(builder, value, typ, inner_type)
    imp = context.get_getattr(inner_type, attr)
    return imp(context, builder, inner_type, val, attr)

@lower_cast(types.Any, types.DeferredType)
@lower_cast(types.Optional, types.DeferredType)
@lower_cast(types.Boolean, types.DeferredType)
def any_to_deferred(context, builder, fromty, toty, val):
    actual = context.cast(builder, val, fromty, toty.get())
    model = context.data_model_manager[toty]
    return model.set(builder, model.make_uninitialized(), actual)

@lower_cast(types.DeferredType, types.Any)
@lower_cast(types.DeferredType, types.Boolean)
@lower_cast(types.DeferredType, types.Optional)
def deferred_to_any(context, builder, fromty, toty, val):
    model = context.data_model_manager[fromty]
    val = model.get(builder, val)
    return context.cast(builder, val, fromty.get(), toty)


#------------------------------------------------------------------------------

@lower_builtin(operator.getitem, types.CPointer, types.Integer)
def getitem_cpointer(context, builder, sig, args):
    base_ptr, idx = args
    elem_ptr = builder.gep(base_ptr, [idx])
    res = builder.load(elem_ptr)
    return impl_ret_borrowed(context, builder, sig.return_type, res)


@lower_builtin(operator.setitem, types.CPointer, types.Integer, types.Any)
def setitem_cpointer(context, builder, sig, args):
    base_ptr, idx, val = args
    elem_ptr = builder.gep(base_ptr, [idx])
    builder.store(val, elem_ptr)


#-------------------------------------------------------------------------------

def do_minmax(context, builder, argtys, args, cmpop):
    assert len(argtys) == len(args), (argtys, args)
    assert len(args) > 0

    def binary_minmax(accumulator, value):
        # This is careful to reproduce Python's algorithm, e.g.
        # max(1.5, nan, 2.5) should return 2.5 (not nan or 1.5)
        accty, acc = accumulator
        vty, v = value
        ty = context.typing_context.unify_types(accty, vty)
        assert ty is not None
        acc = context.cast(builder, acc, accty, ty)
        v = context.cast(builder, v, vty, ty)
        cmpsig = typing.signature(types.boolean, ty, ty)
        ge = context.get_function(cmpop, cmpsig)
        pred = ge(builder, (v, acc))
        res = builder.select(pred, v, acc)
        return ty, res

    typvals = zip(argtys, args)
    resty, resval = reduce(binary_minmax, typvals)
    return resval


@lower_builtin(max, types.BaseTuple)
def max_iterable(context, builder, sig, args):
    argtys = list(sig.args[0])
    args = cgutils.unpack_tuple(builder, args[0])
    return do_minmax(context, builder, argtys, args, operator.gt)

@lower_builtin(max, types.VarArg(types.Any))
def max_vararg(context, builder, sig, args):
    return do_minmax(context, builder, sig.args, args, operator.gt)

@lower_builtin(min, types.BaseTuple)
def min_iterable(context, builder, sig, args):
    argtys = list(sig.args[0])
    args = cgutils.unpack_tuple(builder, args[0])
    return do_minmax(context, builder, argtys, args, operator.lt)

@lower_builtin(min, types.VarArg(types.Any))
def min_vararg(context, builder, sig, args):
    return do_minmax(context, builder, sig.args, args, operator.lt)


def _round_intrinsic(tp):
    # round() rounds half to even on Python 3, away from zero on Python 2.
    if utils.IS_PY3:
        return "llvm.rint.f%d" % (tp.bitwidth,)
    else:
        return "llvm.round.f%d" % (tp.bitwidth,)

@lower_builtin(round, types.Float)
def round_impl_unary(context, builder, sig, args):
    fltty = sig.args[0]
    llty = context.get_value_type(fltty)
    module = builder.module
    fnty = Type.function(llty, [llty])
    fn = module.get_or_insert_function(fnty, name=_round_intrinsic(fltty))
    res = builder.call(fn, args)
    if utils.IS_PY3:
        # unary round() returns an int on Python 3
        res = builder.fptosi(res, context.get_value_type(sig.return_type))
    return impl_ret_untracked(context, builder, sig.return_type, res)

@lower_builtin(round, types.Float, types.Integer)
def round_impl_binary(context, builder, sig, args):
    fltty = sig.args[0]
    # Allow calling the intrinsic from the Python implementation below.
    # This avoids the conversion to an int in Python 3's unary round().
    _round = types.ExternalFunction(
        _round_intrinsic(fltty), typing.signature(fltty, fltty))

    def round_ndigits(x, ndigits):
        if math.isinf(x) or math.isnan(x):
            return x

        if ndigits >= 0:
            if ndigits > 22:
                # pow1 and pow2 are each safe from overflow, but
                # pow1*pow2 ~= pow(10.0, ndigits) might overflow.
                pow1 = 10.0 ** (ndigits - 22)
                pow2 = 1e22
            else:
                pow1 = 10.0 ** ndigits
                pow2 = 1.0
            y = (x * pow1) * pow2
            if math.isinf(y):
                return x
            return (_round(y) / pow2) / pow1

        else:
            pow1 = 10.0 ** (-ndigits)
            y = x / pow1
            return _round(y) * pow1

    res = context.compile_internal(builder, round_ndigits, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


#-------------------------------------------------------------------------------
# Numeric constructors

@lower_builtin(int, types.Any)
@lower_builtin(float, types.Any)
def int_impl(context, builder, sig, args):
    [ty] = sig.args
    [val] = args
    res = context.cast(builder, val, ty, sig.return_type)
    return impl_ret_untracked(context, builder, sig.return_type, res)


@lower_builtin(complex, types.VarArg(types.Any))
def complex_impl(context, builder, sig, args):
    complex_type = sig.return_type
    float_type = complex_type.underlying_float
    if len(sig.args) == 1:
        [argty] = sig.args
        [arg] = args
        if isinstance(argty, types.Complex):
            # Cast Complex* to Complex*
            res = context.cast(builder, arg, argty, complex_type)
            return impl_ret_untracked(context, builder, sig.return_type, res)
        else:
            real = context.cast(builder, arg, argty, float_type)
            imag = context.get_constant(float_type, 0)

    elif len(sig.args) == 2:
        [realty, imagty] = sig.args
        [real, imag] = args
        real = context.cast(builder, real, realty, float_type)
        imag = context.cast(builder, imag, imagty, float_type)

    cmplx = context.make_complex(builder, complex_type)
    cmplx.real = real
    cmplx.imag = imag
    res = cmplx._getvalue()
    return impl_ret_untracked(context, builder, sig.return_type, res)


@lower_builtin(types.NumberClass, types.Any)
@lower_builtin(types.TypeRef, types.Any)
def number_constructor(context, builder, sig, args):
    """
    Call a number class, e.g. np.int32(...)
    """
    if isinstance(sig.return_type, types.Array):
        # Array constructor
        impl = context.get_function(np.array, sig)
        return impl(builder, args)
    else:
        # Scalar constructor
        [val] = args
        [valty] = sig.args
        return context.cast(builder, val, valty, sig.return_type)


#-------------------------------------------------------------------------------
# Constants

@lower_constant(types.Dummy)
def constant_dummy(context, builder, ty, pyval):
    # This handles None, etc.
    return context.get_dummy_value()

@lower_constant(types.ExternalFunctionPointer)
def constant_function_pointer(context, builder, ty, pyval):
    ptrty = context.get_function_pointer_type(ty)
    ptrval = context.add_dynamic_addr(builder, ty.get_pointer(pyval),
                                      info=str(pyval))
    return builder.bitcast(ptrval, ptrty)


@lower_constant(types.Optional)
def constant_optional(context, builder, ty, pyval):
    if pyval is None:
        return context.make_optional_none(builder, ty.type)
    else:
        return context.make_optional_value(builder, ty.type, pyval)


# -----------------------------------------------------------------------------

@lower_builtin(type, types.Any)
def type_impl(context, builder, sig, args):
    """
    One-argument type() builtin.
    """
    return context.get_dummy_value()


@lower_builtin(iter, types.IterableType)
def iter_impl(context, builder, sig, args):
    ty, = sig.args
    val, = args
    iterval = call_getiter(context, builder, ty, val)
    return iterval


@lower_builtin(next, types.IteratorType)
def next_impl(context, builder, sig, args):
    iterty, = sig.args
    iterval, = args

    res = call_iternext(context, builder, iterty, iterval)

    with builder.if_then(builder.not_(res.is_valid()), likely=False):
        context.call_conv.return_user_exc(builder, StopIteration, ())

    return res.yielded_value()


# -----------------------------------------------------------------------------

@lower_builtin("not in", types.Any, types.Any)
def not_in(context, builder, sig, args):
    def in_impl(a, b):
        return operator.contains(b, a)

    res = context.compile_internal(builder, in_impl, sig, args)
    return builder.not_(res)


# -----------------------------------------------------------------------------

@lower_builtin(len, types.ConstSized)
def constsized_len(context, builder, sig, args):
    [ty] = sig.args
    retty = sig.return_type
    res = context.get_constant(retty, len(ty.types))
    return impl_ret_untracked(context, builder, sig.return_type, res)


@lower_builtin(bool, types.Sized)
def sized_bool(context, builder, sig, args):
    [ty] = sig.args
    if len(ty):
        return cgutils.true_bit
    else:
        return cgutils.false_bit

@lower_builtin(tuple)
def lower_empty_tuple(context, builder, sig, args):
    retty = sig.return_type
    res = context.get_constant_undef(retty)
    return impl_ret_untracked(context, builder, sig.return_type, res)

@lower_builtin(tuple, types.BaseTuple)
def lower_tuple(context, builder, sig, args):
    val, = args
    return impl_ret_untracked(context, builder, sig.return_type, val)

@overload(bool)
def bool_sequence(x):
    valid_types = (
        types.CharSeq,
        types.UnicodeCharSeq,
        types.DictType,
        types.ListType,
        types.UnicodeType,
        types.Set,
    )
    
    if isinstance(x, valid_types):
        def bool_impl(x):
            return len(x) > 0
        return bool_impl

# -----------------------------------------------------------------------------

def get_type_max_value(typ):
    if isinstance(typ, types.Float):
        bw = typ.bitwidth
        if bw == 32:
            return np.finfo(np.float32).max
        if bw == 64:
            return np.finfo(np.float64).max
        raise NotImplementedError("Unsupported floating point type")
    if isinstance(typ, types.Integer):
        return typ.maxval
    raise NotImplementedError("Unsupported type")

def get_type_min_value(typ):
    if isinstance(typ, types.Float):
        bw = typ.bitwidth
        if bw == 32:
            return np.finfo(np.float32).min
        if bw == 64:
            return np.finfo(np.float64).min
        raise NotImplementedError("Unsupported floating point type")
    if isinstance(typ, types.Integer):
        return typ.minval
    raise NotImplementedError("Unsupported type")

@lower_builtin(get_type_min_value, types.NumberClass)
@lower_builtin(get_type_min_value, types.DType)
def lower_get_type_min_value(context, builder, sig, args):
    typ = sig.args[0].dtype
    bw = typ.bitwidth

    if isinstance(typ, types.Integer):
        lty = ir.IntType(bw)
        val = typ.minval
        res = ir.Constant(lty, val)
    elif isinstance(typ, types.Float):
        if bw == 32:
            lty = ir.FloatType()
        elif bw == 64:
            lty = ir.DoubleType()
        else:
            raise NotImplementedError("llvmlite only supports 32 and 64 bit floats")
        npty = getattr(np, 'float{}'.format(bw))
        res = ir.Constant(lty, np.finfo(npty).min)
    return impl_ret_untracked(context, builder, lty, res)

@lower_builtin(get_type_max_value, types.NumberClass)
@lower_builtin(get_type_max_value, types.DType)
def lower_get_type_max_value(context, builder, sig, args):
    typ = sig.args[0].dtype
    bw = typ.bitwidth

    if isinstance(typ, types.Integer):
        lty = ir.IntType(bw)
        val = typ.maxval
        res = ir.Constant(lty, val)
    elif isinstance(typ, types.Float):
        if bw == 32:
            lty = ir.FloatType()
        elif bw == 64:
            lty = ir.DoubleType()
        else:
            raise NotImplementedError("llvmlite only supports 32 and 64 bit floats")
        npty = getattr(np, 'float{}'.format(bw))
        res = ir.Constant(lty, np.finfo(npty).max)
    return impl_ret_untracked(context, builder, lty, res)

# -----------------------------------------------------------------------------

from numba.typing.builtins import IndexValue, IndexValueType
from numba.extending import overload, register_jitable

@lower_builtin(IndexValue, types.intp, types.Type)
@lower_builtin(IndexValue, types.uintp, types.Type)
def impl_index_value(context, builder, sig, args):
    typ = sig.return_type
    index, value = args
    index_value = cgutils.create_struct_proxy(typ)(context, builder)
    index_value.index = index
    index_value.value = value
    return index_value._getvalue()


@overload(min)
def indval_min(indval1, indval2):
    if isinstance(indval1, IndexValueType) and \
       isinstance(indval2, IndexValueType):
        def min_impl(indval1, indval2):
            if indval1.value > indval2.value:
                return indval2
            return indval1
        return min_impl


@overload(max)
def indval_max(indval1, indval2):
    if isinstance(indval1, IndexValueType) and \
       isinstance(indval2, IndexValueType):
        def max_impl(indval1, indval2):
            if indval2.value > indval1.value:
                return indval2
            return indval1
        return max_impl


greater_than = register_jitable(lambda a, b: a > b)
less_than = register_jitable(lambda a, b: a < b)


@register_jitable
def min_max_impl(iterable, op):
    if isinstance(iterable, types.IterableType):
        def impl(iterable):
            it = iter(iterable)
            return_val = next(it)
            for val in it:
                if op(val, return_val):
                    return_val = val
            return return_val
        return impl


@overload(min)
def iterable_min(iterable):
    return min_max_impl(iterable, less_than)


@overload(max)
def iterable_max(iterable):
    return min_max_impl(iterable, greater_than)


@lower_builtin(types.TypeRef)
def redirect_type_ctor(context, builder, sig, args):
    """Redirect constructor implementation to `numba_typeref_ctor(cls, *args)`,
    which should be overloaded by type implementator.

    For example:

        d = Dict()

    `d` will be typed as `TypeRef[DictType]()`.  Thus, it will call into this
    implementation.  We need to redirect the lowering to a function
    named ``numba_typeref_ctor``.
    """
    cls = sig.return_type

    def call_ctor(cls, *args):
        return numba_typeref_ctor(cls, *args)

    # Pack arguments into a tuple for `*args`
    ctor_args = types.Tuple.from_types(sig.args)
    # Make signature T(TypeRef[T], *args) where T is cls
    sig = typing.signature(cls, types.TypeRef(cls), ctor_args)

    args = (context.get_dummy_value(),   # Type object has no runtime repr.
            context.make_tuple(builder, sig.args[1], args))

    return context.compile_internal(builder, call_ctor, sig, args)

# ------------------------------------------------------------------------------
# map, filter, reduce


@overload(map)
def ol_map(func, iterable, *args):
    def impl(func, iterable, *args):
        for x in zip(iterable, *args):
            yield func(*x)
    return impl


@overload(filter)
def ol_filter(func, iterable):
    if (func is None) or isinstance(func, types.NoneType):
        def impl(func, iterable):
            for x in iterable:
                if x:
                    yield x
    else:
        def impl(func, iterable):
            for x in iterable:
                if func(x):
                    yield x
    return impl
