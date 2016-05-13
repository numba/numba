from __future__ import print_function, absolute_import, division

import math
from functools import reduce

import numpy as np

from llvmlite import ir
from llvmlite.llvmpy.core import Type, Constant
import llvmlite.llvmpy.core as lc

from .imputils import (lower_builtin, lower_getattr, lower_getattr_generic,
                       lower_cast, lower_constant, iternext_impl,
                       call_getiter, call_iternext,
                       impl_ret_borrowed, impl_ret_untracked)
from .. import typing, types, cgutils, utils


@lower_builtin('is not', types.Any, types.Any)
def generic_is_not(context, builder, sig, args):
    """
    Implement `x is not y` as `not (x is y)`.
    """
    is_impl = context.get_function('is', sig)
    return builder.not_(is_impl(builder, args))


@lower_builtin('is', types.Any, types.Any)
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
                    eq_impl = context.get_function('==', sig)
                except NotImplementedError:
                    # no `==` implemented for this type
                    return cgutils.false_bit
                else:
                    return eq_impl(builder, args)
    else:
        return cgutils.false_bit

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

@lower_builtin('getitem', types.CPointer, types.Integer)
def getitem_cpointer(context, builder, sig, args):
    base_ptr, idx = args
    elem_ptr = builder.gep(base_ptr, [idx])
    res = builder.load(elem_ptr)
    return impl_ret_borrowed(context, builder, sig.return_type, res)


@lower_builtin('setitem', types.CPointer, types.Integer, types.Any)
def setitem_cpointer(context, builder, sig, args):
    base_ptr, idx, val = args
    elem_ptr = builder.gep(base_ptr, [idx])
    builder.store(val, elem_ptr)


#-------------------------------------------------------------------------------

@lower_builtin(max, types.VarArg(types.Any))
def max_impl(context, builder, sig, args):
    argtys = sig.args
    for a in argtys:
        if a not in types.number_domain:
            raise AssertionError("only implemented for numeric types")

    def domax(a, b):
        at, av = a
        bt, bv = b
        ty = context.typing_context.unify_types(at, bt)
        assert ty is not None
        cav = context.cast(builder, av, at, ty)
        cbv = context.cast(builder, bv, bt, ty)
        cmpsig = typing.signature(types.boolean, ty, ty)
        ge = context.get_function(">=", cmpsig)
        pred = ge(builder, (cav, cbv))
        res = builder.select(pred, cav, cbv)
        return ty, res

    typvals = zip(argtys, args)
    resty, resval = reduce(domax, typvals)
    return impl_ret_borrowed(context, builder, sig.return_type, resval)


@lower_builtin(min, types.VarArg(types.Any))
def min_impl(context, builder, sig, args):
    argtys = sig.args
    for a in argtys:
        if a not in types.number_domain:
            raise AssertionError("only implemented for numeric types")

    def domax(a, b):
        at, av = a
        bt, bv = b
        ty = context.typing_context.unify_types(at, bt)
        assert ty is not None
        cav = context.cast(builder, av, at, ty)
        cbv = context.cast(builder, bv, bt, ty)
        cmpsig = typing.signature(types.boolean, ty, ty)
        le = context.get_function("<=", cmpsig)
        pred = le(builder, (cav, cbv))
        res = builder.select(pred, cav, cbv)
        return ty, res

    typvals = zip(argtys, args)
    resty, resval = reduce(domax, typvals)
    return impl_ret_borrowed(context, builder, sig.return_type, resval)


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
    ptrval = context.get_constant_generic(builder, types.intp,
                                          ty.get_pointer(pyval))
    return builder.inttoptr(ptrval, ptrty)


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
        return a in b

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
