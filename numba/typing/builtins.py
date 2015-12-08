from __future__ import print_function, division, absolute_import

import itertools

from numba import types, intrinsics
from numba.utils import PYVERSION, RANGE_ITER_OBJECTS, operator_map
from numba.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                    AbstractTemplate, builtin_global, builtin,
                                    builtin_attr, signature, bound_function,
                                    make_callable_template)

for obj in RANGE_ITER_OBJECTS:
    builtin_global(obj, types.range_type)
builtin_global(len, types.len_type)
builtin_global(slice, types.slice_type)
builtin_global(abs, types.abs_type)
builtin_global(print, types.print_type)


@builtin
class Print(ConcreteTemplate):
    key = types.print_type
    intcases = [signature(types.none, ty) for ty in types.integer_domain]
    realcases = [signature(types.none, ty) for ty in types.real_domain]
    cases = intcases + realcases


@builtin
class PrintOthers(AbstractTemplate):
    key = types.print_type

    def accepted_types(self, ty):
        if ty in types.integer_domain or ty in types.real_domain:
            return True

        if isinstance(ty, types.CharSeq):
            return True

    def generic(self, args, kws):
        assert not kws, "kwargs to print is not supported."
        for a in args:
            if not self.accepted_types(a):
                raise TypeError("Type %s is not printable." % a)
        return signature(types.none, *args)


@builtin
class Abs(ConcreteTemplate):
    key = types.abs_type
    int_cases = [signature(ty, ty) for ty in types.signed_domain]
    real_cases = [signature(ty, ty) for ty in types.real_domain]
    complex_cases = [signature(ty.underlying_float, ty)
                     for ty in types.complex_domain]
    cases = int_cases + real_cases + complex_cases


@builtin
class Slice(ConcreteTemplate):
    key = types.slice_type
    cases = [
        signature(types.slice3_type),
        signature(types.slice3_type, types.none, types.none),
        signature(types.slice3_type, types.none, types.intp),
        signature(types.slice3_type, types.intp, types.none),
        signature(types.slice3_type, types.intp, types.intp),
        signature(types.slice3_type, types.intp, types.intp, types.intp),
        signature(types.slice3_type, types.none, types.intp, types.intp),
        signature(types.slice3_type, types.intp, types.none, types.intp),
        signature(types.slice3_type, types.none, types.none, types.intp),
    ]


@builtin
class Range(ConcreteTemplate):
    key = types.range_type
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


@builtin
class GetIter(AbstractTemplate):
    key = "getiter"

    def generic(self, args, kws):
        assert not kws
        [obj] = args
        if isinstance(obj, types.IterableType):
            return signature(obj.iterator_type, obj)


@builtin
class IterNext(AbstractTemplate):
    key = "iternext"

    def generic(self, args, kws):
        assert not kws
        [it] = args
        if isinstance(it, types.IteratorType):
            return signature(types.Pair(it.yield_type, types.boolean), it)


@builtin
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


@builtin
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


@builtin
class BinOpAdd(BinOp):
    key = "+"


@builtin
class BinOpSub(BinOp):
    key = "-"


@builtin
class BinOpMul(BinOp):
    key = "*"


@builtin
class BinOpDiv(BinOp):
    key = "/?"


@builtin
class BinOpMod(ConcreteTemplate):
    key = "%"
    cases = list(integer_binop_cases)
    cases += [signature(op, op, op) for op in sorted(types.real_domain)]


@builtin
class BinOpTrueDiv(ConcreteTemplate):
    key = "/"
    cases = [signature(types.float64, op1, op2)
             for op1, op2 in itertools.product(machine_ints, machine_ints)]
    cases += [signature(op, op, op) for op in sorted(types.real_domain)]
    cases += [signature(op, op, op) for op in sorted(types.complex_domain)]


@builtin
class BinOpFloorDiv(ConcreteTemplate):
    key = "//"
    cases = list(integer_binop_cases)
    cases += [signature(op, op, op) for op in sorted(types.real_domain)]


@builtin
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


class PowerBuiltin(BinOpPower):
    key = pow
    # TODO add 3 operand version

builtin_global(pow, types.Function(PowerBuiltin))


class BitwiseShiftOperation(ConcreteTemplate):
    cases = list(integer_binop_cases)


@builtin
class BitwiseLeftShift(BitwiseShiftOperation):
    key = "<<"


@builtin
class BitwiseRightShift(BitwiseShiftOperation):
    key = ">>"


class BitwiseLogicOperation(BinOp):
    cases = list(integer_binop_cases)


@builtin
class BitwiseAnd(BitwiseLogicOperation):
    key = "&"


@builtin
class BitwiseOr(BitwiseLogicOperation):
    key = "|"


@builtin
class BitwiseXor(BitwiseLogicOperation):
    key = "^"


# Bitwise invert and negate are special: we must not upcast the operand
# for unsigned numbers, as that would change the result.
# (i.e. ~np.int8(0) == 255 but ~np.int32(0) == 4294967295).

@builtin
class BitwiseInvert(ConcreteTemplate):
    key = "~"

    cases = [signature(types.int8, types.boolean)]
    cases += [signature(choose_result_int(op), op) for op in types.unsigned_domain]
    cases += [signature(choose_result_int(op), op) for op in types.signed_domain]


class UnaryOp(ConcreteTemplate):
    cases = [signature(choose_result_int(op), op) for op in types.unsigned_domain]
    cases += [signature(choose_result_int(op), op) for op in types.signed_domain]
    cases += [signature(op, op) for op in sorted(types.real_domain)]
    cases += [signature(op, op) for op in sorted(types.complex_domain)]


@builtin
class UnaryNegate(UnaryOp):
    key = "-"


@builtin
class UnaryPositive(UnaryOp):
    key = "+"


@builtin
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


@builtin
class CmpOpLt(OrderedCmpOp):
    key = '<'

@builtin
class CmpOpLe(OrderedCmpOp):
    key = '<='

@builtin
class CmpOpGt(OrderedCmpOp):
    key = '>'

@builtin
class CmpOpGe(OrderedCmpOp):
    key = '>='

@builtin
class CmpOpEq(UnorderedCmpOp):
    key = '=='

@builtin
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

@builtin
class TupleEq(TupleCompare):
    key = '=='

@builtin
class TupleNe(TupleCompare):
    key = '!='

@builtin
class TupleGe(TupleCompare):
    key = '>='

@builtin
class TupleGt(TupleCompare):
    key = '>'

@builtin
class TupleLe(TupleCompare):
    key = '<='

@builtin
class TupleLt(TupleCompare):
    key = '<'


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
        builtin(template)


class CmpOpIdentity(AbstractTemplate):
    def generic(self, args, kws):
        [lhs, rhs] = args
        return signature(types.boolean, lhs, rhs)


@builtin
class CmpOpIs(CmpOpIdentity):
    key = 'is'


@builtin
class CmpOpIsNot(CmpOpIdentity):
    key = 'is not'


def normalize_1d_index(index):
    """
    Normalize the *index* type (an integer or slice) for indexing a 1D
    sequence.
    """
    if index == types.slice3_type:
        return types.slice3_type

    elif isinstance(index, types.Integer):
        return types.intp if index.signed else types.uintp


@builtin
class GetItemCPointer(AbstractTemplate):
    key = "getitem"

    def generic(self, args, kws):
        assert not kws
        ptr, idx = args
        if isinstance(ptr, types.CPointer) and isinstance(idx, types.Integer):
            return signature(ptr.dtype, ptr, normalize_1d_index(idx))


@builtin
class SetItemCPointer(AbstractTemplate):
    key = "setitem"

    def generic(self, args, kws):
        assert not kws
        ptr, idx, val = args
        if isinstance(ptr, types.CPointer) and isinstance(idx, types.Integer):
            return signature(types.none, ptr, normalize_1d_index(idx), ptr.dtype)


@builtin
class Len(AbstractTemplate):
    key = types.len_type

    def generic(self, args, kws):
        assert not kws
        (val,) = args
        if isinstance(val, (types.Buffer, types.BaseTuple)):
            return signature(types.intp, val)


@builtin
class TupleBool(AbstractTemplate):
    key = "is_true"

    def generic(self, args, kws):
        assert not kws
        (val,) = args
        if isinstance(val, (types.BaseTuple)):
            return signature(types.boolean, val)


@builtin
class StaticGetItemTuple(AbstractTemplate):
    key = "static_getitem"

    def generic(self, args, kws):
        tup, idx = args
        if isinstance(tup, types.BaseTuple) and isinstance(idx, int):
            return tup.types[idx]


#-------------------------------------------------------------------------------

@builtin_attr
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


@builtin_attr
class BooleanAttribute(AttributeTemplate):
    key = types.Boolean

    def resolve___class__(self, ty):
        return types.NumberClass(ty)


@builtin_attr
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


@builtin_attr
class SliceAttribute(AttributeTemplate):
    key = types.slice3_type

    def resolve_start(self, ty):
        return types.intp

    def resolve_stop(self, ty):
        return types.intp

    def resolve_step(self, ty):
        return types.intp


#-------------------------------------------------------------------------------


@builtin_attr
class NumberClassAttribute(AttributeTemplate):
    key = types.NumberClass

    def resolve___call__(self, classty):
        """
        Resolve a number class's constructor (e.g. calling int(...))
        """
        ty = classty.instance_type

        def typer(val):
            return ty

        return types.Function(make_callable_template(key=ty, typer=typer))


def register_number_classes(register_global):
    nb_types = set(types.number_domain)
    nb_types.add(types.bool_)

    for ty in nb_types:
        register_global(ty, types.NumberClass(ty))


register_number_classes(builtin_global)


#------------------------------------------------------------------------------


class Max(AbstractTemplate):
    key = max

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


class Min(AbstractTemplate):
    key = min

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


class Round(ConcreteTemplate):
    key = round
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


builtin_global(max, types.Function(Max))
builtin_global(min, types.Function(Min))
builtin_global(round, types.Function(Round))


#------------------------------------------------------------------------------


class Bool(AbstractTemplate):
    key = bool

    def generic(self, args, kws):
        assert not kws
        [arg] = args
        if isinstance(arg, (types.Boolean, types.Number)):
            return signature(types.boolean, arg)
        # XXX typing for bool cannot be polymorphic because of the
        # types.Function thing, so we redirect to the "is_true"
        # intrinsic.
        return self.context.resolve_function_type("is_true", args, kws)


class Int(AbstractTemplate):
    key = int

    def generic(self, args, kws):
        assert not kws

        [arg] = args

        if isinstance(arg, types.Integer):
            return signature(arg, arg)
        if isinstance(arg, (types.Float, types.Boolean)):
            return signature(types.intp, arg)


class Float(AbstractTemplate):
    key = float

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


class Complex(AbstractTemplate):
    key = complex

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


builtin_global(bool, types.Function(Bool))
builtin_global(int, types.Function(Int))
builtin_global(float, types.Function(Float))
builtin_global(complex, types.Function(Complex))


#------------------------------------------------------------------------------

@builtin
class Enumerate(AbstractTemplate):
    key = enumerate

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


builtin_global(enumerate, types.Function(Enumerate))


@builtin
class Zip(AbstractTemplate):
    key = zip

    def generic(self, args, kws):
        assert not kws
        if all(isinstance(it, types.IterableType) for it in args):
            zip_type = types.ZipType(args)
            return signature(zip_type, *args)


builtin_global(zip, types.Function(Zip))


@builtin
class Intrinsic_array_ravel(AbstractTemplate):
    key = intrinsics.array_ravel

    def generic(self, args, kws):
        assert not kws
        [arr] = args
        if arr.layout in 'CF' and arr.ndim >= 1:
            return signature(arr.copy(ndim=1), arr)

builtin_global(intrinsics.array_ravel, types.Function(Intrinsic_array_ravel))


#------------------------------------------------------------------------------

@builtin
class TypeBuiltin(AbstractTemplate):
    key = type

    def generic(self, args, kws):
        assert not kws
        if len(args) == 1:
            # One-argument type() -> return the __class__
            try:
                classty = self.context.resolve_getattr(args[0], "__class__")
            except KeyError:
                return
            else:
                return signature(classty, *args)


builtin_global(type, types.Function(TypeBuiltin))
