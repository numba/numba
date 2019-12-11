from __future__ import print_function, division, absolute_import

import itertools

import numpy as np
import operator

from numba import types, prange, errors
from numba.parfor import internal_prange

from numba.utils import PYVERSION, RANGE_ITER_OBJECTS, IS_PY3
from numba.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                    AbstractTemplate, infer_global, infer,
                                    infer_getattr, signature, bound_function,
                                    make_callable_template)


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
    int_cases = [signature(ty, ty) for ty in sorted(types.signed_domain)]
    uint_cases = [signature(ty, ty) for ty in sorted(types.unsigned_domain)]
    real_cases = [signature(ty, ty) for ty in sorted(types.real_domain)]
    complex_cases = [signature(ty.underlying_float, ty)
                     for ty in sorted(types.complex_domain)]
    cases = int_cases + uint_cases +  real_cases + complex_cases


@infer_global(slice)
class Slice(ConcreteTemplate):
    cases = [
        signature(types.slice2_type, types.intp),
        signature(types.slice2_type, types.none),
        signature(types.slice2_type, types.none, types.none),
        signature(types.slice2_type, types.none, types.intp),
        signature(types.slice2_type, types.intp, types.none),
        signature(types.slice2_type, types.intp, types.intp),
        signature(types.slice3_type, types.intp, types.intp, types.intp),
        signature(types.slice3_type, types.none, types.intp, types.intp),
        signature(types.slice3_type, types.intp, types.none, types.intp),
        signature(types.slice3_type, types.intp, types.intp, types.none),
        signature(types.slice3_type, types.intp, types.none, types.none),
        signature(types.slice3_type, types.none, types.intp, types.none),
        signature(types.slice3_type, types.none, types.none, types.intp),
        signature(types.slice3_type, types.none, types.none, types.none),
    ]


class Range(ConcreteTemplate):
    cases = [
        signature(types.range_state32_type, types.int32),
        signature(types.range_state32_type, types.int32, types.int32),
        signature(types.range_state32_type, types.int32, types.int32,
                  types.int32),
        signature(types.range_state64_type, types.int64),
        signature(types.range_state64_type, types.int64, types.int64),
        signature(types.range_state64_type, types.int64, types.int64,
                  types.int64),
        signature(types.unsigned_range_state64_type, types.uint64),
        signature(types.unsigned_range_state64_type, types.uint64, types.uint64),
        signature(types.unsigned_range_state64_type, types.uint64, types.uint64,
                  types.uint64),
    ]

for func in RANGE_ITER_OBJECTS:
    infer_global(func, typing_key=range)(Range)

infer_global(prange, typing_key=prange)(Range)
infer_global(internal_prange, typing_key=internal_prange)(Range)

@infer
class GetIter(AbstractTemplate):
    key = "getiter"

    def generic(self, args, kws):
        assert not kws
        [obj] = args
        if isinstance(obj, types.IterableType):
            # Raise this here to provide a very specific message about this
            # common issue, delaying the error until later leads to something
            # less specific being noted as the problem (e.g. no support for
            # getiter on array(<>, 2, 'C')).
            if isinstance(obj, types.Array) and obj.ndim > 1:
                msg = ("Direct iteration is not supported for arrays with "
                       "dimension > 1. Try using indexing instead.")
                raise errors.TypingError(msg)
            else:
                return signature(obj.iterator_type, obj)


@infer
class IterNext(AbstractTemplate):
    key = "iternext"

    def generic(self, args, kws):
        assert not kws
        [it] = args
        if isinstance(it, types.IteratorType):
            return signature(types.Pair(it.yield_type, types.boolean), it)


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


def choose_result_bitwidth(*inputs):
    return max(types.intp.bitwidth, *(tp.bitwidth for tp in inputs))

def choose_result_int(*inputs):
    """
    Choose the integer result type for an operation on integer inputs,
    according to the integer typing NBEP.
    """
    bitwidth = choose_result_bitwidth(*inputs)
    signed = any(tp.signed for tp in inputs)
    return types.Integer.from_bitwidth(bitwidth, signed)


# The "machine" integer types to take into consideration for operator typing
# (according to the integer typing NBEP)
machine_ints = (
    sorted(set((types.intp, types.int64))) +
    sorted(set((types.uintp, types.uint64)))
    )

# Explicit integer rules for binary operators; smaller ints will be
# automatically upcast.
integer_binop_cases = tuple(
    signature(choose_result_int(op1, op2), op1, op2)
    for op1, op2 in itertools.product(machine_ints, machine_ints)
    )


class BinOp(ConcreteTemplate):
    cases = list(integer_binop_cases)
    cases += [signature(op, op, op) for op in sorted(types.real_domain)]
    cases += [signature(op, op, op) for op in sorted(types.complex_domain)]


@infer_global(operator.add)
class BinOpAdd(BinOp):
    pass


@infer_global(operator.iadd)
class BinOpAdd(BinOp):
    pass


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

if not IS_PY3:
    @infer_global(operator.div)
    class BinOpDiv(BinOp):
        pass


    @infer_global(operator.idiv)
    class BinOpDiv(BinOp):
        pass


@infer_global(operator.mod)
class BinOpMod(ConcreteTemplate):
    cases = list(integer_binop_cases)
    cases += [signature(op, op, op) for op in sorted(types.real_domain)]


@infer_global(operator.imod)
class BinOpMod(ConcreteTemplate):
    cases = list(integer_binop_cases)
    cases += [signature(op, op, op) for op in sorted(types.real_domain)]


@infer_global(operator.truediv)
class BinOpTrueDiv(ConcreteTemplate):
    cases = [signature(types.float64, op1, op2)
             for op1, op2 in itertools.product(machine_ints, machine_ints)]
    cases += [signature(op, op, op) for op in sorted(types.real_domain)]
    cases += [signature(op, op, op) for op in sorted(types.complex_domain)]


@infer_global(operator.itruediv)
class BinOpTrueDiv(ConcreteTemplate):
    cases = [signature(types.float64, op1, op2)
             for op1, op2 in itertools.product(machine_ints, machine_ints)]
    cases += [signature(op, op, op) for op in sorted(types.real_domain)]
    cases += [signature(op, op, op) for op in sorted(types.complex_domain)]


@infer_global(operator.floordiv)
class BinOpFloorDiv(ConcreteTemplate):
    cases = list(integer_binop_cases)
    cases += [signature(op, op, op) for op in sorted(types.real_domain)]


@infer_global(operator.ifloordiv)
class BinOpFloorDiv(ConcreteTemplate):
    cases = list(integer_binop_cases)
    cases += [signature(op, op, op) for op in sorted(types.real_domain)]


@infer_global(divmod)
class DivMod(ConcreteTemplate):
    _tys = machine_ints + sorted(types.real_domain)
    cases = [signature(types.UniTuple(ty, 2), ty, ty) for ty in _tys]


@infer_global(operator.pow)
class BinOpPower(ConcreteTemplate):
    cases = list(integer_binop_cases)
    # Ensure that float32 ** int doesn't go through DP computations
    cases += [signature(types.float32, types.float32, op)
              for op in (types.int32, types.int64, types.uint64)]
    cases += [signature(types.float64, types.float64, op)
              for op in (types.int32, types.int64, types.uint64)]
    cases += [signature(op, op, op)
              for op in sorted(types.real_domain)]
    cases += [signature(op, op, op)
              for op in sorted(types.complex_domain)]


@infer_global(operator.ipow)
class BinOpPower(ConcreteTemplate):
    cases = list(integer_binop_cases)
    # Ensure that float32 ** int doesn't go through DP computations
    cases += [signature(types.float32, types.float32, op)
              for op in (types.int32, types.int64, types.uint64)]
    cases += [signature(types.float64, types.float64, op)
              for op in (types.int32, types.int64, types.uint64)]
    cases += [signature(op, op, op)
              for op in sorted(types.real_domain)]
    cases += [signature(op, op, op)
              for op in sorted(types.complex_domain)]


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
    cases = [signature(max(op, types.intp), op, op2)
             for op in sorted(types.signed_domain)
             for op2 in [types.uint64, types.int64]]
    cases += [signature(max(op, types.uintp), op, op2)
              for op in sorted(types.unsigned_domain)
              for op2 in [types.uint64, types.int64]]
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
    cases = [signature(types.boolean, types.boolean, types.boolean)]
    cases += list(integer_binop_cases)
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
    cases = [signature(types.boolean, types.boolean)]
    cases += [signature(choose_result_int(op), op) for op in sorted(types.unsigned_domain)]
    cases += [signature(choose_result_int(op), op) for op in sorted(types.signed_domain)]

    unsafe_casting = False


class UnaryOp(ConcreteTemplate):
    cases = [signature(choose_result_int(op), op) for op in sorted(types.unsigned_domain)]
    cases += [signature(choose_result_int(op), op) for op in sorted(types.signed_domain)]
    cases += [signature(op, op) for op in sorted(types.real_domain)]
    cases += [signature(op, op) for op in sorted(types.complex_domain)]
    cases += [signature(types.intp, types.boolean)]


@infer_global(operator.neg)
class UnaryNegate(UnaryOp):
    pass


@infer_global(operator.pos)
class UnaryPositive(UnaryOp):
   pass


@infer_global(operator.not_)
class UnaryNot(ConcreteTemplate):
    cases = [signature(types.boolean, types.boolean)]
    cases += [signature(types.boolean, op) for op in sorted(types.signed_domain)]
    cases += [signature(types.boolean, op) for op in sorted(types.unsigned_domain)]
    cases += [signature(types.boolean, op) for op in sorted(types.real_domain)]
    cases += [signature(types.boolean, op) for op in sorted(types.complex_domain)]


class OrderedCmpOp(ConcreteTemplate):
    cases = [signature(types.boolean, types.boolean, types.boolean)]
    cases += [signature(types.boolean, op, op) for op in sorted(types.signed_domain)]
    cases += [signature(types.boolean, op, op) for op in sorted(types.unsigned_domain)]
    cases += [signature(types.boolean, op, op) for op in sorted(types.real_domain)]


class UnorderedCmpOp(ConcreteTemplate):
    cases = OrderedCmpOp.cases + [
        signature(types.boolean, op, op) for op in sorted(types.complex_domain)]


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


@infer_global(operator.eq)
class CmpOpEq(UnorderedCmpOp):
    pass


@infer_global(operator.eq)
class ConstOpEq(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        (arg1, arg2) = args
        if isinstance(arg1, types.Literal) and isinstance(arg2, types.Literal):
            return signature(types.boolean, arg1, arg2)


@infer_global(operator.ne)
class ConstOpNotEq(ConstOpEq):
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
                return signature(types.boolean, lhs, rhs)


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
        [lhs, rhs] = args
        return signature(types.boolean, lhs, rhs)


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
        return types.intp if index.signed else types.uintp


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
            return signature(types.intp, val)
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
            return signature(types.boolean, seq, val)

@infer_global(operator.truth)
class TupleBool(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        (val,) = args
        if isinstance(val, (types.BaseTuple)):
            return signature(types.boolean, val)


@infer
class StaticGetItemTuple(AbstractTemplate):
    key = "static_getitem"

    def generic(self, args, kws):
        tup, idx = args
        if not isinstance(tup, types.BaseTuple):
            return
        if isinstance(idx, int):
            ret = tup.types[idx]
        elif isinstance(idx, slice):
            ret = types.BaseTuple.from_types(tup.types[idx])
        return signature(ret, *args)


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

    if PYVERSION >= (3,):
        def resolve_contiguous(self, buf):
            return types.boolean

        def resolve_c_contiguous(self, buf):
            return types.boolean

        def resolve_f_contiguous(self, buf):
            return types.boolean

    def resolve_itemsize(self, buf):
        return types.intp

    def resolve_nbytes(self, buf):
        return types.intp

    def resolve_readonly(self, buf):
        return types.boolean

    def resolve_shape(self, buf):
        return types.UniTuple(types.intp, buf.ndim)

    def resolve_strides(self, buf):
        return types.UniTuple(types.intp, buf.ndim)

    def resolve_ndim(self, buf):
        return types.intp


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
class SliceAttribute(AttributeTemplate):
    key = types.SliceType

    def resolve_start(self, ty):
        return types.intp

    def resolve_stop(self, ty):
        return types.intp

    def resolve_step(self, ty):
        return types.intp

    @bound_function("slice.indices")
    def resolve_indices(self, ty, args, kws):
        assert not kws
        if len(args) != 1:
            raise TypeError(
                "indices() takes exactly one argument (%d given)" % len(args)
            )
        typ, = args
        if not isinstance(typ, types.Integer):
            raise TypeError(
                "'%s' object cannot be interpreted as an integer" % typ
            )
        return signature(types.UniTuple(types.intp, 3), types.intp)


#-------------------------------------------------------------------------------


@infer_getattr
class NumberClassAttribute(AttributeTemplate):
    key = types.NumberClass

    def resolve___call__(self, classty):
        """
        Resolve a number class's constructor (e.g. calling int(...))
        """
        ty = classty.instance_type

        def typer(val):
            if isinstance(val, (types.BaseTuple, types.Sequence)):
                # Array constructor, e.g. np.int32([1, 2])
                sig = self.context.resolve_function_type(
                    np.array, (val,), {'dtype': types.DType(ty)})
                return sig.return_type
            else:
                # Scalar constructor, e.g. np.int32(42)
                return ty

        return types.Function(make_callable_template(key=ty, typer=typer))


@infer_getattr
class TypeRefAttribute(AttributeTemplate):
    key = types.TypeRef

    def resolve___call__(self, classty):
        """
        Resolve a number class's constructor (e.g. calling int(...))

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
            def redirect(*args, **kwargs):
                return self.context.resolve_function_type(ty, args, kwargs)
            return types.Function(make_callable_template(key=ty, typer=redirect))


#------------------------------------------------------------------------------


class MinMaxBase(AbstractTemplate):

    def _unify_minmax(self, tys):
        for ty in tys:
            if not isinstance(ty, types.Number):
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
    if PYVERSION < (3, 0):
        cases = [
            signature(types.float32, types.float32),
            signature(types.float64, types.float64),
        ]
    else:
        cases = [
            signature(types.intp, types.float32),
            signature(types.int64, types.float64),
        ]
    cases += [
        signature(types.float32, types.float32, types.intp),
        signature(types.float64, types.float64, types.intp),
    ]


#------------------------------------------------------------------------------


@infer_global(bool)
class Bool(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        [arg] = args
        if isinstance(arg, (types.Boolean, types.Number)):
            return signature(types.boolean, arg)
        # XXX typing for bool cannot be polymorphic because of the
        # types.Function thing, so we redirect to the operator.truth
        # intrinsic.
        return self.context.resolve_function_type(operator.truth, args, kws)


@infer_global(int)
class Int(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws

        [arg] = args

        if isinstance(arg, types.Integer):
            return signature(arg, arg)
        if isinstance(arg, (types.Float, types.Boolean)):
            return signature(types.intp, arg)


@infer_global(float)
class Float(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws

        [arg] = args

        if arg not in types.number_domain:
            raise TypeError("float() only support for numbers")

        if arg in types.complex_domain:
            raise TypeError("float() does not support complex")

        if arg in types.integer_domain:
            return signature(types.float64, arg)

        elif arg in types.real_domain:
            return signature(arg, arg)


@infer_global(complex)
class Complex(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws

        if len(args) == 1:
            [arg] = args
            if arg not in types.number_domain:
                raise TypeError("complex() only support for numbers")
            if arg == types.float32:
                return signature(types.complex64, arg)
            else:
                return signature(types.complex128, arg)

        elif len(args) == 2:
            [real, imag] = args
            if (real not in types.number_domain or
                imag not in types.number_domain):
                raise TypeError("complex() only support for numbers")
            if real == imag == types.float32:
                return signature(types.complex64, real, imag)
            else:
                return signature(types.complex128, real, imag)


#------------------------------------------------------------------------------

@infer_global(enumerate)
class Enumerate(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        it = args[0]
        if len(args) > 1 and not isinstance(args[1], types.Integer):
            raise TypeError("Only integers supported as start value in "
                            "enumerate")
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
            classty = self.context.resolve_getattr(args[0], "__class__")
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

from numba.targets.builtins import get_type_min_value, get_type_max_value

@infer_global(get_type_min_value)
@infer_global(get_type_max_value)
class MinValInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        assert isinstance(args[0], (types.DType, types.NumberClass))
        return signature(args[0].dtype, *args)


#------------------------------------------------------------------------------

from numba.extending import (
    typeof_impl, type_callable, models, register_model, make_attribute_wrapper,
    )


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
        if ind == types.intp or ind == types.uintp:
            return IndexValueType(mval)
    return typer


@register_model(IndexValueType)
class IndexValueModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('index', types.intp),
            ('value', fe_type.val_typ),
            ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(IndexValueType, 'index', 'index')
make_attribute_wrapper(IndexValueType, 'value', 'value')
