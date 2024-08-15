import itertools

import numpy as np
import operator
import math

from numba.core import types, errors, config
from numba import prange
from numba.parfors.parfor import internal_prange

from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                         AbstractTemplate, infer_global, infer,
                                         infer_getattr, signature,
                                         bound_function, make_callable_template)


from numba.core.extending import (
    typeof_impl, type_callable, models, register_model, make_attribute_wrapper, overload, intrinsic, overload_method
    )
from numba.cpython.new_numbers import real_add_impl, int_add_impl, complex_add_impl

register_model(types.NotImplementedType)(models.OpaqueModel)


@infer_global(print)
class Print(AbstractTemplate):
    def generic(self, args, kws):
        for a in args:
            sig = self.context.resolve_function_type("print_item", (a,), {})
            if sig is None:
                raise TypeError("Type %s is not printable." % a)
            assert sig.return_type is types.none
        return signature(types.none, *args)

@infer
class PrintItem(AbstractTemplate):
    key = "print_item"

    def generic(self, args, kws):
        arg, = args
        return signature(types.none, *args)


@infer_global(abs)
class Abs(ConcreteTemplate):
    int_cases = [signature(ty, ty) for ty in sorted(types.py_signed_domain)]
    real_cases = [signature(ty, ty) for ty in sorted(types.py_real_domain)]
    complex_cases = [signature(ty.underlying_float, ty)
                     for ty in sorted(types.py_complex_domain)]
    cases = int_cases + real_cases + complex_cases


@infer_global(slice)
class Slice(ConcreteTemplate):
    cases = [
        signature(types.slice2_type, types.py_int),
        signature(types.slice2_type, types.none),
        signature(types.slice2_type, types.none, types.none),
        signature(types.slice2_type, types.none, types.py_int),
        signature(types.slice2_type, types.py_int, types.none),
        signature(types.slice2_type, types.py_int, types.py_int),
        signature(types.slice3_type, types.py_int, types.py_int, types.py_int),
        signature(types.slice3_type, types.none, types.py_int, types.py_int),
        signature(types.slice3_type, types.py_int, types.none, types.py_int),
        signature(types.slice3_type, types.py_int, types.py_int, types.none),
        signature(types.slice3_type, types.py_int, types.none, types.none),
        signature(types.slice3_type, types.none, types.py_int, types.none),
        signature(types.slice3_type, types.none, types.none, types.py_int),
        signature(types.slice3_type, types.none, types.none, types.none),
    ]


@infer_global(range, typing_key=range)
@infer_global(prange, typing_key=prange)
@infer_global(internal_prange, typing_key=internal_prange)
class Range(ConcreteTemplate):
    cases = [
        signature(types.range_state_type, types.py_int),
        signature(types.range_state_type, types.py_int, types.py_int),
        signature(types.range_state_type, types.py_int, types.py_int,
                types.py_int),
    ]


@infer
class GetIter(AbstractTemplate):
    key = "getiter"

    def generic(self, args, kws):
        assert not kws
        [obj] = args
        if isinstance(obj, types.IterableType):
            return signature(obj.iterator_type, obj)


@infer
class IterNext(AbstractTemplate):
    key = "iternext"

    def generic(self, args, kws):
        assert not kws
        [it] = args
        if isinstance(it, types.IteratorType):
            return signature(types.Pair(it.yield_type, types.py_bool), it)


@infer
class PairFirst(AbstractTemplate):
    """
    Given a heterogeneous pair, return the first element.
    """
    key = "pair_first"

    def generic(self, args, kws):
        assert not kws
        [pair] = args
        if isinstance(pair, types.Pair):
            return signature(pair.first_type, pair)


@infer
class PairSecond(AbstractTemplate):
    """
    Given a heterogeneous pair, return the second element.
    """
    key = "pair_second"

    def generic(self, args, kws):
        assert not kws
        [pair] = args
        if isinstance(pair, types.Pair):
            return signature(pair.second_type, pair)


class BinOp(ConcreteTemplate):
    cases = []


@overload(operator.is_)
def ol_is_NotImplemented(a, b, /):
    a_is_notimplemented_ty = isinstance(a, types.NotImplementedType)
    b_is_notimplemented_ty = isinstance(b, types.NotImplementedType)

    # only use this overload if at least one of the types is NotImplementedType
    if a_is_notimplemented_ty or b_is_notimplemented_ty:
        pred = a_is_notimplemented_ty and b_is_notimplemented_ty
        def ol_is_NotImplemented(a, b, /):
            return pred
        return ol_is_NotImplemented


@overload(operator.is_not)
def ol_is_not_NotImplemented(a, b, /):
    a_is_notimplemented_ty = isinstance(a, types.NotImplementedType)
    b_is_notimplemented_ty = isinstance(b, types.NotImplementedType)

    # only use this overload if at least one of the types is NotImplementedType
    if a_is_notimplemented_ty or b_is_notimplemented_ty:
        pred = not(a_is_notimplemented_ty and b_is_notimplemented_ty)
        def ol_is_not_NotImplemented(a, b, /):
            return pred
        return ol_is_not_NotImplemented


def bool_add_bool(x, y):
    return NotImplemented


def int_add_int(x, y):
    return NotImplemented


def float_add_float(x, y):
    return NotImplemented


def complex_add_complex(x, y):
    return NotImplemented


@intrinsic
def intrin_bool_add_bool(tyctx, boolxty, boolyty):
    assert boolxty == boolyty
    sig = types.py_int(boolxty, boolyty)
    def codegen(cgctx, builder, sig, llargs):
        new_args = [cgctx.cast(builder, v, t, sig.return_type) for v, t in zip(llargs, sig.args)]
        return builder.add(*new_args)

    return sig, codegen


@intrinsic
def intrin_int_add_int(tyctx, intxty, intyty):
    assert intxty == intyty
    sig = intxty(intxty, intxty)
    def codegen(cgctx, builder, sig, llargs):
        return builder.add(*llargs)

    return sig, codegen


@intrinsic
def intrin_float_add_float(tyctx, floatxty, floatyty):
    assert floatxty == floatyty, f"{floatxty} != {floatyty}"
    sig = floatxty(floatxty, floatyty)
    def codegen(cgctx, builder, sig, llargs):
        return builder.fadd(*llargs)
    return sig, codegen


@intrinsic
def intrin_complex_add_complex(tyctx, compxty, compyty):
    assert compxty == compyty
    sig = compxty(compxty, compyty)
    def codegen(context, builder, sig, args):
        [cx, cy] = args
        ty = sig.args[0]
        x = context.make_complex(builder, ty, value=cx)
        y = context.make_complex(builder, ty, value=cy)
        z = context.make_complex(builder, ty)
        z.real = builder.fadd(x.real, y.real)
        z.imag = builder.fadd(x.imag, y.imag)
        res = z._getvalue()
        return res

    return sig, codegen


@overload(bool_add_bool)
def ol_bool_add_bool(x, y):
    if x == y:
        def impl(x, y):
            return intrin_bool_add_bool(x, y)
        return impl


@overload(int_add_int)
def ol_int_add_int(x, y):
    if x == y:
        def impl(x, y):
            return intrin_int_add_int(x, y)
        return impl


@overload(float_add_float)
def ol_float_add_float(x, y):
    if x == y:
        def impl(x, y):
            return intrin_float_add_float(x, y)
        return impl


@overload(complex_add_complex)
def ol_complex_add_complex(x, y):
    if x == y:
        def impl(x, y):
            return intrin_complex_add_complex(x, y)
        return impl


@overload_method(types.PythonBoolean, "__bool__")
def py_bool__bool__(self):
    def impl(self):
        return self
    return impl


@overload_method(types.PythonBoolean, "__int__")
def py_bool__int__(self):
    def impl(self):
        return types.py_int(self)
    return impl


@overload_method(types.PythonBoolean, "__float__")
def py_bool__float__(self):
    def impl(self):
        return types.py_float(self)
    return impl


@overload_method(types.PythonInteger, "__bool__")
def py_int__bool__(self):
    def impl(self):
        return types.py_bool(self)
    return impl


@overload_method(types.PythonInteger, "__int__")
def py_int__int__(self):
    def impl(self):
        return self
    return impl


@overload_method(types.PythonInteger, "__float__")
def py_int__float__(self):
    def impl(self):
        return types.py_float(self)
    return impl


@overload_method(types.PythonFloat, "__bool__")
def py_float__bool__(self):
    def impl(self):
        return types.py_bool(self)
    return impl


@overload_method(types.PythonFloat, "__int__")
def py_float__int__(self):
    def impl(self):
        return types.py_int(self)
    return impl


@overload_method(types.PythonFloat, "__float__")
def py_float__float__(self):
    def impl(self):
        return self
    return impl


@overload_method(types.PythonComplex, "__complex__")
def py_complex__complex__(self):
    def impl(self):
        return self
    return impl


@overload_method(types.NumPyFloat, "__float__")
def np_float64__complex__(self):
    if self.bitwidth == 64:
        def impl(self):
            return types.py_float(self)
    else:
        def impl(self):
            return NotImplemented
    return impl


@overload_method(types.NumPyFloat, "__complex__")
def np_float64__complex__(self):
    if self.bitwidth == 64:
        def impl(self):
            return types.py_complex(self)
    else:
        def impl(self):
            return NotImplemented
    return impl


@overload_method(types.NumPyComplex, "__complex__")
def np_float64__complex__(self):
    if self.bitwidth == 128:
        def impl(self):
            return types.py_complex(self)
    else:
        def impl(self):
            return NotImplemented
    return impl


@overload_method(types.PythonBoolean, "__add__")
@overload_method(types.PythonBoolean, "__radd__")
def py_bool__add__(self, other):
    def impl(self, other):
        if isinstance(other, bool):
            return bool_add_bool(self, other)
        elif isinstance(other, int):
            return int_add_int(int(self), other)
        else:
            return NotImplemented
    return impl


@overload_method(types.PythonInteger, "__add__")
@overload_method(types.PythonInteger, "__radd__")
def py_int__add__(self, other):
    def impl(self, other):
        if isinstance(other, int):
            return int_add_int(self, other)
        elif isinstance(other, bool):
            return int_add_int(self, int(other))
        else:
            return NotImplemented
    return impl


@overload_method(types.PythonFloat, "__add__")
@overload_method(types.PythonFloat, "__radd__")
def py_float__add__(self, other):
    def impl(self, other):
        if isinstance(other, float):
            return float_add_float(self, other)
        elif isinstance(other, (int, bool, np.float64)):
            return float_add_float(self, float(other))
        else:
            return NotImplemented
    return impl


@overload_method(types.PythonComplex, "__add__")
@overload_method(types.PythonComplex, "__radd__")
def py_complex__add__(self, other):
    def impl(self, other):
        if isinstance(other, complex):
            return complex_add_complex(self, other)
        elif isinstance(other, (float, int, bool, np.float64, np.complex128)):
            return complex_add_complex(self, complex(other))
        else:
            return NotImplemented
    return impl


FROM_DTYPE = {
    np.dtype('bool'): types.np_bool_,
    np.dtype('int8'): types.np_int8,
    np.dtype('int16'): types.np_int16,
    np.dtype('int32'): types.np_int32,
    np.dtype('int64'): types.np_int64,

    np.dtype('uint8'): types.np_uint8,
    np.dtype('uint16'): types.np_uint16,
    np.dtype('uint32'): types.np_uint32,
    np.dtype('uint64'): types.np_uint64,

    np.dtype('float32'): types.np_float32,
    np.dtype('float64'): types.np_float64,
    np.dtype('float16'): types.np_float16,
    np.dtype('complex64'): types.np_complex64,
    np.dtype('complex128'): types.np_complex128,

    np.dtype(object): types.pyobject,
}

binop_cache = {
    "__add__": {},
    "__radd__": {}
}

@intrinsic
def np_bool_add_bool(tyctx, boolxty, boolyty):
    sig = types.np_bool_(boolxty, boolyty)
    def codegen(cgctx, builder, sig, llargs):
        return builder.or_(*llargs)

    return sig, codegen

def find_np_res_type(op, op_cache, argtys):
    if argtys in op_cache[op]:
        return op_cache[op][argtys]

    try:
        argvals = [argty.cast_python_value(0) for argty in argtys]
    except NotImplementedError:
        e = errors.NumbaTypeError(f"cast_python_value({argtys})")
        raise e
    res_val = getattr(argvals[0], op)(*argvals[1:])
    if res_val is not NotImplemented:
        res_val = FROM_DTYPE[res_val.dtype]
    op_cache[op][argtys] = res_val
    return res_val


@overload_method(types.NumPyBoolean, "__add__")
@overload_method(types.NumPyInteger, "__add__")
@overload_method(types.NumPyFloat, "__add__")
@overload_method(types.NumPyComplex, "__add__")
def np__add__(self, other):
    final_ty = find_np_res_type("__add__", binop_cache, (self, other))

    if final_ty is NotImplemented:
        def impl(self, other):
            return NotImplemented
        return impl

    if 'bool' in final_ty.name:
        add_func = np_bool_add_bool
    elif 'int' in final_ty.name:
        add_func = int_add_int
    elif 'float' in final_ty.name:
        add_func = float_add_float
    elif 'complex' in final_ty.name:
        add_func = complex_add_complex
    else:
        raise TypeError(f"Addition function not defined for {self} and {other}")

    # evaluate type
    def impl(self, other):
        return add_func(final_ty(self), final_ty(other))
    return impl


@overload_method(types.NumPyBoolean, "__radd__")
@overload_method(types.NumPyInteger, "__radd__")
@overload_method(types.NumPyFloat, "__radd__")
@overload_method(types.NumPyComplex, "__radd__")
def np__radd__(self, other):
    final_ty = find_np_res_type("__radd__", binop_cache, (self, other))

    if final_ty is NotImplemented:
        def impl(self, other):
            return NotImplemented
        return impl

    if 'bool' in final_ty.name:
        add_func = np_bool_add_bool
    elif 'int' in final_ty.name:
        add_func = int_add_int
    elif 'float' in final_ty.name:
        add_func = float_add_float
    elif 'complex' in final_ty.name:
        add_func = complex_add_complex
    else:
        raise TypeError(f"Addition function not defined for {self} and {other}")

    # evaluate type
    def impl(self, other):
        return add_func(final_ty(self), final_ty(other))
    return impl


def generate_binop(op_func, slot, rslot, opchar):

    # based loosely on:
    # https://github.com/python/cpython/blob/1a1e013a4a526546c373afd887f2e25eecc984ad/Objects/abstract.c#L916-L977
    def binary_op(v, w):
        pass

    @overload(binary_op)
    def ol_binary_op_number(v, w):
        try:
            is_subclass = issubclass(
                type(w.cast_python_value(0)),
                type(v.cast_python_value(0))
            )
        except NotImplementedError:
            is_subclass = False

        def impl(v, w):
            if is_subclass:
                if hasattr(w, rslot):
                    ret = getattr(w, rslot)(v)
                    if ret is not NotImplemented:
                        return ret

            if hasattr(v, slot):
                ret = getattr(v, slot)(w)
                if ret is not NotImplemented:
                    return ret

            if hasattr(w, rslot):
                ret = getattr(w, rslot)(v)
                if ret is not NotImplemented:
                    return ret

            return NotImplemented
        return impl

    def raise_error():
        pass

    @overload(raise_error)
    def ol_raise_error(v, w):
        # this refers to numba types but should probably be mapped back to the
        # types in python
        msg = f"unsupported operand type(s) for {opchar}: {v} and {w}"
        def impl(v, w):
            raise TypeError(msg)
        return impl

    @overload(op_func)
    def binary_op_impl(v, w):
        def impl(v, w):
            result = binary_op(v, w)
            if result is NotImplemented:
                raise_error(v, w)
            else:
                return result

        return impl


generate_binop(operator.add, "__add__", "__radd__", "+")


@infer_global(operator.sub)
class BinOpSub(BinOp):
    pass


@infer_global(operator.isub)
class BinOpSub(BinOp):
    pass


@infer_global(operator.mul)
class BinOpMul(BinOp):
    pass


@infer_global(operator.imul)
class BinOpMul(BinOp):
    pass


@infer_global(operator.mod)
class BinOpMod(ConcreteTemplate):
    cases = []


@infer_global(operator.imod)
class BinOpMod(ConcreteTemplate):
    cases = []


@infer_global(operator.truediv)
class BinOpTrueDiv(ConcreteTemplate):
    cases = []


@infer_global(operator.itruediv)
class BinOpTrueDiv(ConcreteTemplate):
    cases = []


@infer_global(operator.floordiv)
class BinOpFloorDiv(ConcreteTemplate):
    cases = []


@infer_global(operator.ifloordiv)
class BinOpFloorDiv(ConcreteTemplate):
    cases = []


@infer_global(divmod)
class DivMod(ConcreteTemplate):
    # This probably needs a mixture
    _tys = {types.py_int, types.py_float} | types.np_number_domain | types.np_real_domain
    cases = [signature(types.UniTuple(ty, 2), ty, ty) for ty in _tys]


@infer_global(operator.pow)
class BinOpPower(ConcreteTemplate):
    cases = []


@infer_global(operator.ipow)
class BinOpPower(ConcreteTemplate):
    cases = []


@infer_global(pow)
class PowerBuiltin(BinOpPower):
    # TODO add 3 operand version
    pass


class BitwiseShiftOperation(ConcreteTemplate):
    # For bitshifts, only the first operand's signedness matters
    # to choose the operation's signedness (the second operand
    # should always be positive but will generally be considered
    # signed anyway, since it's often a constant integer).
    # (also, see issue #1995 for right-shifts)

    # The RHS type is fixed to 64-bit signed/unsigned ints.
    # The implementation will always cast the operands to the width of the
    # result type, which is the widest between the LHS type and (u)intp.
    cases = [signature(max(op, types.py_int), op, op2)
             for op in types.py_signed_domain
             for op2 in types.py_signed_domain]
    unsafe_casting = False


@infer_global(operator.lshift)
class BitwiseLeftShift(BitwiseShiftOperation):
    pass

@infer_global(operator.ilshift)
class BitwiseLeftShift(BitwiseShiftOperation):
    pass


@infer_global(operator.rshift)
class BitwiseRightShift(BitwiseShiftOperation):
    pass


@infer_global(operator.irshift)
class BitwiseRightShift(BitwiseShiftOperation):
    pass


class BitwiseLogicOperation(BinOp):
    cases = []
    unsafe_casting = False


@infer_global(operator.and_)
class BitwiseAnd(BitwiseLogicOperation):
    pass


@infer_global(operator.iand)
class BitwiseAnd(BitwiseLogicOperation):
    pass


@infer_global(operator.or_)
class BitwiseOr(BitwiseLogicOperation):
    pass


@infer_global(operator.ior)
class BitwiseOr(BitwiseLogicOperation):
    pass


@infer_global(operator.xor)
class BitwiseXor(BitwiseLogicOperation):
    pass


@infer_global(operator.ixor)
class BitwiseXor(BitwiseLogicOperation):
    pass


# Bitwise invert and negate are special: we must not upcast the operand
# for unsigned numbers, as that would change the result.
# (i.e. ~np.int8(0) == 255 but ~np.int32(0) == 4294967295).

@infer_global(operator.invert)
class BitwiseInvert(ConcreteTemplate):
    # Note Numba follows the Numpy semantics of returning a bool,
    # while Python returns an int.  This makes it consistent with
    # np.invert() and makes array expressions correct.
    cases = [signature(types.py_bool, types.py_bool)]
    cases = [signature(types.np_bool_, types.np_bool_)]

    cases += [signature(op, op) for op in sorted(types.np_unsigned_domain)]

    cases += [signature(op, op) for op in sorted(types.py_signed_domain)]
    cases += [signature(op, op) for op in sorted(types.np_signed_domain)]

    unsafe_casting = False


class UnaryOp(ConcreteTemplate):
    cases = [signature(op, op) for op in sorted(types.np_unsigned_domain)]

    cases += [signature(op, op) for op in sorted(types.py_signed_domain)]
    cases += [signature(op, op) for op in sorted(types.np_signed_domain)]

    cases += [signature(op, op) for op in sorted(types.py_real_domain)]
    cases += [signature(op, op) for op in sorted(types.np_real_domain)]

    cases += [signature(op, op) for op in sorted(types.py_complex_domain)]
    cases += [signature(op, op) for op in sorted(types.np_complex_domain)]

    cases += [signature(types.py_int, types.py_bool)]
    cases += [signature(types.np_intp, types.np_bool_)]


@infer_global(operator.neg)
class UnaryNegate(UnaryOp):
    pass


@infer_global(operator.pos)
class UnaryPositive(UnaryOp):
   pass


@infer_global(operator.not_)
class UnaryNot(ConcreteTemplate):
    cases = [signature(types.py_bool, types.py_bool)]
    cases += [signature(types.py_bool, op) for op in sorted(types.py_signed_domain)]
    cases += [signature(types.py_bool, op) for op in sorted(types.py_real_domain)]
    cases += [signature(types.py_bool, op) for op in sorted(types.py_complex_domain)]
    
    cases = [signature(types.np_bool_, types.np_bool_)]
    cases += [signature(types.np_bool_, op) for op in sorted(types.np_signed_domain)]
    cases += [signature(types.np_bool_, op) for op in sorted(types.np_unsigned_domain)]
    cases += [signature(types.np_bool_, op) for op in sorted(types.np_real_domain)]
    cases += [signature(types.np_bool_, op) for op in sorted(types.np_complex_domain)]


class OrderedCmpOp(ConcreteTemplate):
    cases = [signature(types.py_bool, types.py_bool, types.py_bool)]
    cases += [signature(types.py_bool, op, op) for op in sorted(types.py_signed_domain)]
    cases += [signature(types.py_bool, op, op) for op in sorted(types.py_real_domain)]
    cases = [signature(types.np_bool_, types.np_bool_, types.np_bool_)]
    cases += [signature(types.np_bool_, op, op) for op in sorted(types.np_signed_domain)]
    cases += [signature(types.np_bool_, op, op) for op in sorted(types.np_unsigned_domain)]
    cases += [signature(types.np_bool_, op, op) for op in sorted(types.np_real_domain)]


class UnorderedCmpOp(ConcreteTemplate):
    cases = OrderedCmpOp.cases + [
        signature(types.py_bool, op, op) for op in sorted(types.py_complex_domain)] + [
        signature(types.np_bool_, op, op) for op in sorted(types.np_complex_domain)]


@infer_global(operator.lt)
class CmpOpLt(OrderedCmpOp):
    pass


@infer_global(operator.le)
class CmpOpLe(OrderedCmpOp):
    pass


@infer_global(operator.gt)
class CmpOpGt(OrderedCmpOp):
    pass


@infer_global(operator.ge)
class CmpOpGe(OrderedCmpOp):
    pass


# more specific overloads should be registered first
@infer_global(operator.eq)
class ConstOpEq(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        (arg1, arg2) = args
        if isinstance(arg1, types.Literal) and isinstance(arg2, types.Literal):
            return signature(types.np_bool_, arg1, arg2)


@infer_global(operator.ne)
class ConstOpNotEq(ConstOpEq):
    pass


@infer_global(operator.eq)
class CmpOpEq(UnorderedCmpOp):
    pass


@infer_global(operator.ne)
class CmpOpNe(UnorderedCmpOp):
    pass


class TupleCompare(AbstractTemplate):
    def generic(self, args, kws):
        [lhs, rhs] = args
        if isinstance(lhs, types.BaseTuple) and isinstance(rhs, types.BaseTuple):
            for u, v in zip(lhs, rhs):
                # Check element-wise comparability
                res = self.context.resolve_function_type(self.key, (u, v), {})
                if res is None:
                    break
            else:
                return signature(types.py_bool, lhs, rhs)


@infer_global(operator.eq)
class TupleEq(TupleCompare):
    pass


@infer_global(operator.ne)
class TupleNe(TupleCompare):
    pass


@infer_global(operator.ge)
class TupleGe(TupleCompare):
    pass


@infer_global(operator.gt)
class TupleGt(TupleCompare):
    pass


@infer_global(operator.le)
class TupleLe(TupleCompare):
    pass


@infer_global(operator.lt)
class TupleLt(TupleCompare):
    pass


@infer_global(operator.add)
class TupleAdd(AbstractTemplate):
    def generic(self, args, kws):
        if len(args) == 2:
            a, b = args
            if (isinstance(a, types.BaseTuple) and isinstance(b, types.BaseTuple)
                and not isinstance(a, types.BaseNamedTuple)
                and not isinstance(b, types.BaseNamedTuple)):
                res = types.BaseTuple.from_types(tuple(a) + tuple(b))
                return signature(res, a, b)


class CmpOpIdentity(AbstractTemplate):
    def generic(self, args, kws):
        if "NotImplemented" in [str(x) for x in args]:
            return None
        [lhs, rhs] = args
        return signature(types.py_bool, lhs, rhs)


@infer_global(operator.is_)
class CmpOpIs(CmpOpIdentity):
    pass


@infer_global(operator.is_not)
class CmpOpIsNot(CmpOpIdentity):
    pass


def normalize_1d_index(index):
    """
    Normalize the *index* type (an integer or slice) for indexing a 1D
    sequence.
    """
    if isinstance(index, types.SliceType):
        return index

    elif isinstance(index, types.Integer):
        return types.np_intp if index.signed else types.uintp


@infer_global(operator.getitem)
class GetItemCPointer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        ptr, idx = args
        if isinstance(ptr, types.CPointer) and isinstance(idx, types.Integer):
            return signature(ptr.dtype, ptr, normalize_1d_index(idx))


@infer_global(operator.setitem)
class SetItemCPointer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        ptr, idx, val = args
        if isinstance(ptr, types.CPointer) and isinstance(idx, types.Integer):
            return signature(types.none, ptr, normalize_1d_index(idx), ptr.dtype)


@infer_global(len)
class Len(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        (val,) = args
        if isinstance(val, (types.Buffer, types.BaseTuple)):
            return signature(types.py_int, val)
        elif isinstance(val, (types.RangeType)):
            return signature(val.dtype, val)

@infer_global(tuple)
class TupleConstructor(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        # empty tuple case
        if len(args) == 0:
            return signature(types.Tuple(()))
        (val,) = args
        # tuple as input
        if isinstance(val, types.BaseTuple):
            return signature(val, val)


@infer_global(operator.contains)
class Contains(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        (seq, val) = args

        if isinstance(seq, (types.Sequence)):
            return signature(types.py_bool, seq, val)

@infer_global(operator.truth)
class TupleBool(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        (val,) = args
        if isinstance(val, (types.BaseTuple)):
            return signature(types.py_bool, val)


@infer
class StaticGetItemTuple(AbstractTemplate):
    key = "static_getitem"

    def generic(self, args, kws):
        tup, idx = args
        ret = None
        if not isinstance(tup, types.BaseTuple):
            return
        if isinstance(idx, int):
            try:
                ret = tup.types[idx]
            except IndexError:
                raise errors.NumbaIndexError("tuple index out of range")
        elif isinstance(idx, slice):
            ret = types.BaseTuple.from_types(tup.types[idx])
        if ret is not None:
            sig = signature(ret, *args)
            return sig


@infer
class StaticGetItemLiteralList(AbstractTemplate):
    key = "static_getitem"

    def generic(self, args, kws):
        tup, idx = args
        ret = None
        if not isinstance(tup, types.LiteralList):
            return
        if isinstance(idx, int):
            ret = tup.types[idx]
        if ret is not None:
            sig = signature(ret, *args)
            return sig


@infer
class StaticGetItemLiteralStrKeyDict(AbstractTemplate):
    key = "static_getitem"

    def generic(self, args, kws):
        tup, idx = args
        ret = None
        if not isinstance(tup, types.LiteralStrKeyDict):
            return
        if isinstance(idx, str):
            if idx in tup.fields:
                lookup = tup.fields.index(idx)
            else:
                raise errors.NumbaKeyError(f"Key '{idx}' is not in dict.")
            ret = tup.types[lookup]
        if ret is not None:
            sig = signature(ret, *args)
            return sig

@infer
class StaticGetItemClass(AbstractTemplate):
    """This handles the "static_getitem" when a Numba type is subscripted e.g:
    var = typed.List.empty_list(float64[::1, :])
    It only allows this on simple numerical types. Compound types, like
    records, are not supported.
    """
    key = "static_getitem"

    def generic(self, args, kws):
        clazz, idx = args
        if not isinstance(clazz, types.NumberClass):
            return
        ret = clazz.dtype[idx]
        sig = signature(ret, *args)
        return sig


# Generic implementation for "not in"

@infer
class GenericNotIn(AbstractTemplate):
    key = "not in"

    def generic(self, args, kws):
        args = args[::-1]
        sig = self.context.resolve_function_type(operator.contains, args, kws)
        return signature(sig.return_type, *sig.args[::-1])


#-------------------------------------------------------------------------------

@infer_getattr
class MemoryViewAttribute(AttributeTemplate):
    key = types.MemoryView

    def resolve_contiguous(self, buf):
        return types.py_bool

    def resolve_c_contiguous(self, buf):
        return types.py_bool

    def resolve_f_contiguous(self, buf):
        return types.py_bool

    def resolve_itemsize(self, buf):
        return types.py_int

    def resolve_nbytes(self, buf):
        return types.py_int

    def resolve_readonly(self, buf):
        return types.py_bool

    def resolve_shape(self, buf):
        return types.UniTuple(types.py_int, buf.ndim)

    def resolve_strides(self, buf):
        return types.UniTuple(types.py_int, buf.ndim)

    def resolve_ndim(self, buf):
        return types.py_int


#-------------------------------------------------------------------------------


@infer_getattr
class BooleanAttribute(AttributeTemplate):
    key = types.Boolean

    def resolve___class__(self, ty):
        return types.NumberClass(ty)

    @bound_function("number.item")
    def resolve_item(self, ty, args, kws):
        assert not kws
        if not args:
            return signature(ty)


@infer_getattr
class NumberAttribute(AttributeTemplate):
    key = types.Number

    def resolve___class__(self, ty):
        return types.NumberClass(ty)

    def resolve_real(self, ty):
        return getattr(ty, "underlying_float", ty)

    def resolve_imag(self, ty):
        return getattr(ty, "underlying_float", ty)

    @bound_function("complex.conjugate")
    def resolve_conjugate(self, ty, args, kws):
        assert not args
        assert not kws
        return signature(ty)

    @bound_function("number.item")
    def resolve_item(self, ty, args, kws):
        assert not kws
        if not args:
            return signature(ty)


@infer_getattr
class NPTimedeltaAttribute(AttributeTemplate):
    key = types.NPTimedelta

    def resolve___class__(self, ty):
        return types.NumberClass(ty)


@infer_getattr
class NPDatetimeAttribute(AttributeTemplate):
    key = types.NPDatetime

    def resolve___class__(self, ty):
        return types.NumberClass(ty)


@infer_getattr
class SliceAttribute(AttributeTemplate):
    key = types.SliceType

    def resolve_start(self, ty):
        return types.py_int

    def resolve_stop(self, ty):
        return types.py_int

    def resolve_step(self, ty):
        return types.py_int

    @bound_function("slice.indices")
    def resolve_indices(self, ty, args, kws):
        assert not kws
        if len(args) != 1:
            raise errors.NumbaTypeError(
                "indices() takes exactly one argument (%d given)" % len(args)
            )
        typ, = args
        if not isinstance(typ, types.Integer):
            raise errors.NumbaTypeError(
                "'%s' object cannot be interpreted as an integer" % typ
            )
        return signature(types.UniTuple(types.py_int, 3), types.py_int)


#-------------------------------------------------------------------------------


@infer_getattr
class NumberClassAttribute(AttributeTemplate):
    key = types.NumberClass

    def resolve___call__(self, classty):
        """
        Resolve a NumPy number class's constructor (e.g. calling numpy.int32(...))
        """
        ty = classty.instance_type

        def typer(val):
            if isinstance(val, (types.BaseTuple, types.Sequence)):
                # Array constructor, e.g. np.int32([1, 2])
                fnty = self.context.resolve_value_type(np.array)
                sig = fnty.get_call_type(self.context, (val, types.DType(ty)),
                                         {})
                return sig.return_type
            elif isinstance(val, (types.Number, types.Boolean, types.IntEnumMember)):
                 # Scalar constructor, e.g. np.int32(42)
                 return ty
            elif isinstance(val, (types.NPDatetime, types.NPTimedelta)):
                # Constructor cast from datetime-like, e.g.
                # > np.int64(np.datetime64("2000-01-01"))
                if ty.bitwidth == 64:
                    return ty
                else:
                    msg = (f"Cannot cast {val} to {ty} as {ty} is not 64 bits "
                           "wide.")
                    raise errors.TypingError(msg)
            else:
                if (isinstance(val, types.Array) and val.ndim == 0 and
                    val.dtype == ty):
                    # This is 0d array -> scalar degrading
                    return ty
                else:
                    # unsupported
                    msg = f"Casting {val} to {ty} directly is unsupported."
                    if isinstance(val, types.Array):
                        # array casts are supported a different way.
                        msg += f" Try doing '<array>.astype(np.{ty})' instead"
                    raise errors.TypingError(msg)

        return types.Function(make_callable_template(key=ty, typer=typer))


@infer_getattr
class TypeRefAttribute(AttributeTemplate):
    key = types.TypeRef

    def resolve___call__(self, classty):
        """
        Resolve a core number's constructor (e.g. calling int(...))

        Note:

        This is needed because of the limitation of the current type-system
        implementation.  Specifically, the lack of a higher-order type
        (i.e. passing the ``DictType`` vs ``DictType(key_type, value_type)``)
        """
        ty = classty.instance_type

        if isinstance(ty, type) and issubclass(ty, types.Type):
            # Redirect the typing to a:
            #   @type_callable(ty)
            #   def typeddict_call(context):
            #        ...
            # For example, see numba/typed/typeddict.py
            #   @type_callable(DictType)
            #   def typeddict_call(context):
            class Redirect(object):

                def __init__(self, context):
                    self.context =  context

                def __call__(self, *args, **kwargs):
                    result = self.context.resolve_function_type(ty, args, kwargs)
                    if hasattr(result, "pysig"):
                        self.pysig = result.pysig
                    return result

            return types.Function(make_callable_template(key=ty,
                                                         typer=Redirect(self.context)))


#------------------------------------------------------------------------------


class MinMaxBase(AbstractTemplate):

    def _unify_minmax(self, tys):
        for ty in tys:
            if not isinstance(ty, (types.Number, types.NPDatetime, types.NPTimedelta)):
                return
        return self.context.unify_types(*tys)

    def generic(self, args, kws):
        """
        Resolve a min() or max() call.
        """
        assert not kws

        if not args:
            return
        if len(args) == 1:
            # max(arg) only supported if arg is an iterable
            if isinstance(args[0], types.BaseTuple):
                tys = list(args[0])
                if not tys:
                    raise TypeError("%s() argument is an empty tuple"
                                    % (self.key.__name__,))
            else:
                return
        else:
            # max(*args)
            tys = args
        retty = self._unify_minmax(tys)
        if retty is not None:
            return signature(retty, *args)


@infer_global(max)
class Max(MinMaxBase):
    pass


@infer_global(min)
class Min(MinMaxBase):
    pass


@infer_global(round)
class Round(ConcreteTemplate):
    cases = [
        signature(types.py_int, types.py_float),
        signature(types.py_float, types.py_float, types.py_int)
    ]


#------------------------------------------------------------------------------


@infer_global(bool)
class Bool(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        [arg] = args
        if isinstance(arg, (types.Boolean, types.Number)):
            return signature(types.py_bool, arg)
        # XXX typing for bool cannot be polymorphic because of the
        # types.Function thing, so we redirect to the operator.truth
        # intrinsic.
        return self.context.resolve_function_type(operator.truth, args, kws)


@infer_global(int)
class Int(AbstractTemplate):

    def generic(self, args, kws):
        if kws:
            raise errors.NumbaAssertionError('kws not supported')

        [arg] = args

        if isinstance(arg, (types.Integer, types.Float,
                            types.Boolean, types.NPDatetime,
                            types.NPTimedelta)):
            return signature(types.py_int, arg)


@infer_global(float)
class Float(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws

        [arg] = args

        if arg not in (types.py_number_domain | types.np_number_domain | frozenset([types.py_bool, types. np_bool])):
            raise errors.NumbaTypeError("float() only support for numbers")

        if arg in (types.py_complex_domain | types.np_complex_domain):
            raise errors.NumbaTypeError("float() does not support complex")

        return signature(types.py_float, arg)



@infer_global(complex)
class Complex(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        number_domain = types.py_number_domain | types.np_number_domain | frozenset([types.py_bool, types.np_bool])

        if len(args) == 1:
            [arg] = args
            if arg not in number_domain:
                raise errors.NumbaTypeError("complex() only support for numbers")

            return signature(types.py_complex, arg)

        elif len(args) == 2:
            [real, imag] = args
            if (real not in number_domain or
                imag not in number_domain):
                raise errors.NumbaTypeError("complex() only support for numbers")

            return signature(types.py_complex, real, imag)


#------------------------------------------------------------------------------

@infer_global(enumerate)
class Enumerate(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        it = args[0]
        if len(args) > 1 and not isinstance(args[1], types.Integer):
            raise errors.NumbaTypeError("Only integers supported as start "
                                        "value in enumerate")
        elif len(args) > 2:
            #let python raise its own error
            enumerate(*args)

        if isinstance(it, types.IterableType):
            enumerate_type = types.EnumerateType(it)
            return signature(enumerate_type, *args)


@infer_global(zip)
class Zip(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        if all(isinstance(it, types.IterableType) for it in args):
            zip_type = types.ZipType(args)
            return signature(zip_type, *args)


@infer_global(iter)
class Iter(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        if len(args) == 1:
            it = args[0]
            if isinstance(it, types.IterableType):
                return signature(it.iterator_type, *args)


@infer_global(next)
class Next(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        if len(args) == 1:
            it = args[0]
            if isinstance(it, types.IteratorType):
                return signature(it.yield_type, *args)


#------------------------------------------------------------------------------

@infer_global(type)
class TypeBuiltin(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        if len(args) == 1:
            # One-argument type() -> return the __class__
            # Avoid literal types
            arg = types.unliteral(args[0])
            classty = self.context.resolve_getattr(arg, "__class__")
            if classty is not None:
                return signature(classty, *args)


#------------------------------------------------------------------------------

@infer_getattr
class OptionalAttribute(AttributeTemplate):
    key = types.Optional

    def generic_resolve(self, optional, attr):
        return self.context.resolve_getattr(optional.type, attr)

#------------------------------------------------------------------------------

@infer_getattr
class DeferredAttribute(AttributeTemplate):
    key = types.DeferredType

    def generic_resolve(self, deferred, attr):
        return self.context.resolve_getattr(deferred.get(), attr)

#------------------------------------------------------------------------------


class IndexValue(object):
    """
    Index and value
    """
    def __init__(self, ind, val):
        self.index = ind
        self.value = val

    def __repr__(self):
        return 'IndexValue(%f, %f)' % (self.index, self.value)


class IndexValueType(types.Type):
    def __init__(self, val_typ):
        self.val_typ = val_typ
        super(IndexValueType, self).__init__(
                                    name='IndexValueType({})'.format(val_typ))


@typeof_impl.register(IndexValue)
def typeof_index(val, c):
    val_typ = typeof_impl(val.value, c)
    return IndexValueType(val_typ)


@type_callable(IndexValue)
def type_index_value(context):
    def typer(ind, mval):
        if ind == types.np_intp or ind == types.uintp:
            return IndexValueType(mval)
    return typer


@register_model(IndexValueType)
class IndexValueModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('index', types.np_intp),
            ('value', fe_type.val_typ),
            ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(IndexValueType, 'index', 'index')
make_attribute_wrapper(IndexValueType, 'value', 'value')

