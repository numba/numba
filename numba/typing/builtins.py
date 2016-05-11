from __future__ import print_function, division, absolute_import

import itertools

import numpy as np

from numba import types

from numba.utils import PYVERSION, RANGE_ITER_OBJECTS, operator_map
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

    def is_accepted_type(self, ty):
        if isinstance(ty, (types.Integer, types.Boolean, types.Float,
                           types.CharSeq)):
            return True

    def generic(self, args, kws):
        arg, = args
        if self.is_accepted_type(arg):
            return signature(types.none, *args)


@infer_global(abs)
class Abs(ConcreteTemplate):
    int_cases = [signature(ty, ty) for ty in types.signed_domain]
    real_cases = [signature(ty, ty) for ty in types.real_domain]
    complex_cases = [signature(ty.underlying_float, ty)
                     for ty in types.complex_domain]
    cases = int_cases + real_cases + complex_cases


@infer_global(slice)
class Slice(ConcreteTemplate):
    key = slice
    cases = [
        signature(types.slice2_type),
        signature(types.slice2_type, types.none, types.none),
        signature(types.slice2_type, types.none, types.intp),
        signature(types.slice2_type, types.intp, types.none),
        signature(types.slice2_type, types.intp, types.intp),
        signature(types.slice3_type, types.intp, types.intp, types.intp),
        signature(types.slice3_type, types.none, types.intp, types.intp),
        signature(types.slice3_type, types.intp, types.none, types.intp),
        signature(types.slice3_type, types.none, types.none, types.intp),
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
            return signature(types.Pair(it.yield_type, types.boolean), it)


@infer
class PairFirst(AbstractTemplate):
    """
    Given a heterogenous pair, return the first element.
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
    Given a heterogenous pair, return the second element.
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


@infer
class BinOpAdd(BinOp):
    key = "+"


@infer
class BinOpSub(BinOp):
    key = "-"


@infer
class BinOpMul(BinOp):
    key = "*"


@infer
class BinOpDiv(BinOp):
    key = "/?"


@infer
class BinOpMod(ConcreteTemplate):
    key = "%"
    cases = list(integer_binop_cases)
    cases += [signature(op, op, op) for op in sorted(types.real_domain)]


@infer
class BinOpTrueDiv(ConcreteTemplate):
    key = "/"
    cases = [signature(types.float64, op1, op2)
             for op1, op2 in itertools.product(machine_ints, machine_ints)]
    cases += [signature(op, op, op) for op in sorted(types.real_domain)]
    cases += [signature(op, op, op) for op in sorted(types.complex_domain)]


@infer
class BinOpFloorDiv(ConcreteTemplate):
    key = "//"
    cases = list(integer_binop_cases)
    cases += [signature(op, op, op) for op in sorted(types.real_domain)]


@infer
class BinOpPower(ConcreteTemplate):
    key = "**"
    cases = list(integer_binop_cases)
    cases += [signature(types.float64, types.float64, op)
              for op in sorted(types.signed_domain)]
    cases += [signature(types.float64, types.float64, op)
              for op in sorted(types.unsigned_domain)]
    cases += [signature(op, op, op)
              for op in sorted(types.real_domain)]
    cases += [signature(op, op, op)
              for op in sorted(types.complex_domain)]


@infer_global(pow)
class PowerBuiltin(BinOpPower):
    key = pow
    # TODO add 3 operand version


class BitwiseShiftOperation(ConcreteTemplate):
    cases = list(integer_binop_cases)


@infer
class BitwiseLeftShift(BitwiseShiftOperation):
    key = "<<"


@infer
class BitwiseRightShift(BitwiseShiftOperation):
    key = ">>"


class BitwiseLogicOperation(BinOp):
    cases = [signature(types.boolean, types.boolean, types.boolean)]
    cases += list(integer_binop_cases)


@infer
class BitwiseAnd(BitwiseLogicOperation):
    key = "&"


@infer
class BitwiseOr(BitwiseLogicOperation):
    key = "|"


@infer
class BitwiseXor(BitwiseLogicOperation):
    key = "^"


# Bitwise invert and negate are special: we must not upcast the operand
# for unsigned numbers, as that would change the result.
# (i.e. ~np.int8(0) == 255 but ~np.int32(0) == 4294967295).

@infer
class BitwiseInvert(ConcreteTemplate):
    key = "~"

    # Note Numba follows the Numpy semantics of returning a bool,
    # while Python returns an int.  This makes it consistent with
    # np.invert() and makes array expressions correct.
    cases = [signature(types.boolean, types.boolean)]
    cases += [signature(choose_result_int(op), op) for op in types.unsigned_domain]
    cases += [signature(choose_result_int(op), op) for op in types.signed_domain]


class UnaryOp(ConcreteTemplate):
    cases = [signature(choose_result_int(op), op) for op in types.unsigned_domain]
    cases += [signature(choose_result_int(op), op) for op in types.signed_domain]
    cases += [signature(op, op) for op in sorted(types.real_domain)]
    cases += [signature(op, op) for op in sorted(types.complex_domain)]


@infer
class UnaryNegate(UnaryOp):
    key = "-"


@infer
class UnaryPositive(UnaryOp):
    key = "+"


@infer
class UnaryNot(ConcreteTemplate):
    key = "not"
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


@infer
class CmpOpLt(OrderedCmpOp):
    key = '<'

@infer
class CmpOpLe(OrderedCmpOp):
    key = '<='

@infer
class CmpOpGt(OrderedCmpOp):
    key = '>'

@infer
class CmpOpGe(OrderedCmpOp):
    key = '>='

@infer
class CmpOpEq(UnorderedCmpOp):
    key = '=='

@infer
class CmpOpNe(UnorderedCmpOp):
    key = '!='


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

@infer
class TupleEq(TupleCompare):
    key = '=='

@infer
class TupleNe(TupleCompare):
    key = '!='

@infer
class TupleGe(TupleCompare):
    key = '>='

@infer
class TupleGt(TupleCompare):
    key = '>'

@infer
class TupleLe(TupleCompare):
    key = '<='

@infer
class TupleLt(TupleCompare):
    key = '<'

@infer
class TupleAdd(AbstractTemplate):
    key = '+'

    def generic(self, args, kws):
        if len(args) == 2:
            a, b = args
            if (isinstance(a, types.BaseTuple) and isinstance(b, types.BaseTuple)
                and not isinstance(a, types.BaseNamedTuple)
                and not isinstance(b, types.BaseNamedTuple)):
                res = types.BaseTuple.from_types(tuple(a) + tuple(b))
                return signature(res, a, b)


# Register default implementations of binary inplace operators for
# immutable types.

class InplaceImmutable(AbstractTemplate):
    def generic(self, args, kws):
        lhs, rhs = args
        if not lhs.mutable:
            return self.context.resolve_function_type(self.key[:-1], args, kws)
        # Inplace ops on mutable arguments must be typed explicitly

for _binop, _inp, op in operator_map:
    if _inp:
        template = type('InplaceImmutable_%s' % _binop,
                        (InplaceImmutable,),
                        dict(key=op + '='))
        infer(template)


class CmpOpIdentity(AbstractTemplate):
    def generic(self, args, kws):
        [lhs, rhs] = args
        return signature(types.boolean, lhs, rhs)


@infer
class CmpOpIs(CmpOpIdentity):
    key = 'is'


@infer
class CmpOpIsNot(CmpOpIdentity):
    key = 'is not'


def normalize_1d_index(index):
    """
    Normalize the *index* type (an integer or slice) for indexing a 1D
    sequence.
    """
    if isinstance(index, types.SliceType):
        return index

    elif isinstance(index, types.Integer):
        return types.intp if index.signed else types.uintp


@infer
class GetItemCPointer(AbstractTemplate):
    key = "getitem"

    def generic(self, args, kws):
        assert not kws
        ptr, idx = args
        if isinstance(ptr, types.CPointer) and isinstance(idx, types.Integer):
            return signature(ptr.dtype, ptr, normalize_1d_index(idx))


@infer
class SetItemCPointer(AbstractTemplate):
    key = "setitem"

    def generic(self, args, kws):
        assert not kws
        ptr, idx, val = args
        if isinstance(ptr, types.CPointer) and isinstance(idx, types.Integer):
            return signature(types.none, ptr, normalize_1d_index(idx), ptr.dtype)


@infer_global(len)
class Len(AbstractTemplate):
    key = len

    def generic(self, args, kws):
        assert not kws
        (val,) = args
        if isinstance(val, (types.Buffer, types.BaseTuple)):
            return signature(types.intp, val)


@infer
class TupleBool(AbstractTemplate):
    key = "is_true"

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
            return tup.types[idx]
        elif isinstance(idx, slice):
            return types.BaseTuple.from_types(tup.types[idx])


# Generic implementation for "not in"

@infer
class GenericNotIn(AbstractTemplate):
    key = "not in"

    def generic(self, args, kws):
        return self.context.resolve_function_type("in", args, kws)


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


def register_number_classes(register_global):
    nb_types = set(types.number_domain)
    nb_types.add(types.bool_)

    for ty in nb_types:
        register_global(ty, types.NumberClass(ty))


register_number_classes(infer_global)


#------------------------------------------------------------------------------


@infer_global(max)
class Max(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws

        # max(a, b, ...)
        if len(args) < 2:
            return
        for a in args:
            if a not in types.number_domain:
                return

        retty = self.context.unify_types(*args)
        if retty is not None:
            return signature(retty, *args)


@infer_global(min)
class Min(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws

        # min(a, b, ...)
        if len(args) < 2:
            return
        for a in args:
            if a not in types.number_domain:
                return

        retty = self.context.unify_types(*args)
        if retty is not None:
            return signature(retty, *args)


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


@infer_global(hash)
class Hash(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        arg, = args
        if isinstance(arg, types.Hashable):
            return signature(types.intp, *args)


#------------------------------------------------------------------------------


@infer_global(bool)
class Bool(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        [arg] = args
        if isinstance(arg, (types.Boolean, types.Number)):
            return signature(types.boolean, arg)
        # XXX typing for bool cannot be polymorphic because of the
        # types.Function thing, so we redirect to the "is_true"
        # intrinsic.
        return self.context.resolve_function_type("is_true", args, kws)


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
        if len(args) > 1 and not args[1] in types.integer_domain:
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
