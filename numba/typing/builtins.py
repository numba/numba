from __future__ import print_function, division, absolute_import

from itertools import product

from numba import types
from numba.utils import PYVERSION
from numba.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                    AbstractTemplate, builtin_global, builtin,
                                    builtin_attr, signature)

builtin_global(range, types.range_type)
if PYVERSION < (3, 0):
    builtin_global(xrange, types.range_type)
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


class BinOp(ConcreteTemplate):
    cases = [signature(max(types.intp, op), op, op)
             for op in sorted(types.signed_domain)]
    cases += [signature(max(types.uintp, op), op, op)
              for op in sorted(types.unsigned_domain)]
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
    cases = [signature(op, op, op)
             for op in sorted(types.signed_domain)]
    cases += [signature(op, op, op)
              for op in sorted(types.unsigned_domain)]
    cases += [signature(op, op, op) for op in sorted(types.real_domain)]


@builtin
class BinOpTrueDiv(ConcreteTemplate):
    key = "/"
    cases = [signature(types.float64, op, op)
             for op in sorted(types.signed_domain)]
    cases += [signature(types.float64, op, op)
              for op in sorted(types.unsigned_domain)]
    cases += [signature(op, op, op) for op in sorted(types.real_domain)]
    cases += [signature(op, op, op) for op in sorted(types.complex_domain)]


@builtin
class BinOpFloorDiv(ConcreteTemplate):
    key = "//"
    cases = [signature(op, op, op)
             for op in sorted(types.signed_domain)]
    cases += [signature(op, op, op)
              for op in sorted(types.unsigned_domain)]
    cases += [signature(op, op, op) for op in sorted(types.real_domain)]


@builtin
class BinOpPower(ConcreteTemplate):
    key = "**"
    cases = [signature(types.float64, types.float64, op)
             for op in sorted(types.signed_domain)]
    cases += [signature(types.float64, types.float64, op)
             for op in sorted(types.unsigned_domain)]
    cases += [signature(op, op, op)
             for op in sorted(types.real_domain)]
    cases += [signature(op, op, op)
             for op in sorted(types.complex_domain)]


class BitwiseShiftOperation(ConcreteTemplate):
    cases = [signature(op, op, types.uint32)
             for op in sorted(types.signed_domain)]
    cases += [signature(op, op, types.uint32)
              for op in sorted(types.unsigned_domain)]


@builtin
class BitwiseLeftShift(BitwiseShiftOperation):
    key = "<<"


@builtin
class BitwiseRightShift(BitwiseShiftOperation):
    key = ">>"


class BitwiseLogicOperation(BinOp):
    cases = [signature(max(types.intp, op), op, op)
             for op in sorted(types.signed_domain)]
    cases += [signature(max(types.uintp, op), op, op)
              for op in sorted(types.unsigned_domain)]


@builtin
class BitwiseAnd(BitwiseLogicOperation):
    key = "&"


@builtin
class BitwiseOr(BitwiseLogicOperation):
    key = "|"


@builtin
class BitwiseXor(BitwiseLogicOperation):
    key = "^"


@builtin
class BitwiseInvert(ConcreteTemplate):
    key = "~"

    cases = [signature(types.int8, types.boolean)]
    cases += [signature(op, op) for op in sorted(types.signed_domain)]
    cases += [signature(op, op) for op in sorted(types.unsigned_domain)]


class UnaryOp(ConcreteTemplate):
    cases = [signature(op, op) for op in sorted(types.signed_domain)]
    cases += [signature(op, op) for op in sorted(types.unsigned_domain)]
    cases += [signature(op, op) for op in sorted(types.real_domain)]
    cases += [signature(op, op) for op in sorted(types.complex_domain)]


@builtin
class UnaryNot(UnaryOp):
    key = "not"
    cases = [signature(types.boolean, types.boolean)]
    cases += [signature(types.boolean, op) for op in sorted(types.signed_domain)]
    cases += [signature(types.boolean, op) for op in sorted(types.unsigned_domain)]
    cases += [signature(types.boolean, op) for op in sorted(types.real_domain)]
    cases += [signature(types.boolean, op) for op in sorted(types.complex_domain)]


@builtin
class UnaryNegate(UnaryOp):
    key = "-"
    cases = [signature(max(types.intp, op), op)
             for op in sorted(types.signed_domain)]
    cases += [signature(max(types.uintp, op), op)
              for op in sorted(types.unsigned_domain)]
    cases += [signature(op, op) for op in sorted(types.real_domain)]
    cases += [signature(op, op) for op in sorted(types.complex_domain)]


@builtin
class UnaryPositive(UnaryOp):
    key = "+"
    cases = UnaryNegate.cases


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


def normalize_index(index):
    if isinstance(index, types.UniTuple):
        if index.dtype in types.integer_domain:
            return types.UniTuple(types.intp, len(index))
        elif index.dtype == types.slice3_type:
            return index

    elif isinstance(index, types.Tuple):
        for ty in index:
            if (ty not in types.integer_domain and
                        ty not in types.real_domain and
                        ty != types.slice3_type):
                return
        return index

    elif index == types.slice3_type:
        return types.slice3_type

    # elif index == types.slice2_type:
    #     return types.slice2_type

    else:
        return types.intp


@builtin
class GetItemUniTuple(AbstractTemplate):
    key = "getitem"

    def generic(self, args, kws):
        tup, idx = args
        if isinstance(tup, types.UniTuple):
            return signature(tup.dtype, tup, normalize_index(idx))


@builtin
class GetItemArray(AbstractTemplate):
    key = "getitem"

    def generic(self, args, kws):
        assert not kws
        [ary, idx] = args
        if not isinstance(ary, types.Array):
            return

        idx = normalize_index(idx)
        if not idx:
            return

        if idx == types.slice3_type: #(types.slice2_type, types.slice3_type):
            res = ary.copy(layout='A')
        elif isinstance(idx, (types.UniTuple, types.Tuple)):
            if ary.ndim > len(idx):
                return
            elif ary.ndim < len(idx):
                return
            elif any(i == types.slice3_type for i in idx):
                ndim = ary.ndim
                for i in idx:
                    if i != types.slice3_type:
                        ndim -= 1
                res = ary.copy(ndim=ndim, layout='A')
            else:
                res = ary.dtype
        elif idx == types.intp:
            if ary.ndim != 1:
                return
            res = ary.dtype

        else:
            raise Exception("unreachable: index type of %s" % idx)

        return signature(res, ary, idx)


@builtin
class SetItemArray(AbstractTemplate):
    key = "setitem"

    def generic(self, args, kws):
        assert not kws
        ary, idx, val = args
        if isinstance(ary, types.Array):
            return signature(types.none, ary, normalize_index(idx), ary.dtype)


@builtin
class LenArray(AbstractTemplate):
    key = types.len_type

    def generic(self, args, kws):
        assert not kws
        (ary,) = args
        if isinstance(ary, types.Array):
            return signature(types.intp, ary)

#-------------------------------------------------------------------------------

@builtin_attr
class ArrayAttribute(AttributeTemplate):
    key = types.Array

    def resolve_shape(self, ary):
        return types.UniTuple(types.intp, ary.ndim)

    def resolve_strides(self, ary):
        return types.UniTuple(types.intp, ary.ndim)

    def resolve_ndim(self, ary):
        return types.intp

        #

    # def resolve_flatten(self, ary):
    #     return types.Method(Array_flatten, ary)

    def resolve_size(self, ary):
        return types.intp

    def generic_resolve(self, ary, attr):
        if isinstance(ary.dtype, types.Record):
            if attr in ary.dtype.fields:
                return types.Array(ary.dtype.typeof(attr), ndim=ary.ndim,
                                   layout='A')


class Array_flatten(AbstractTemplate):
    key = "array.flatten"

    def generic(self, args, kws):
        assert not args
        assert not kws
        this = self.this
        if this.layout == 'C':
            resty = this.copy(ndim=1)
            return signature(resty, recvr=this)


@builtin
class CmpOpEqArray(AbstractTemplate):
    key = '=='

    def generic(self, args, kws):
        assert not kws
        [va, vb] = args
        if isinstance(va, types.Array) and va == vb:
            return signature(va.copy(dtype=types.boolean), va, vb)


#-------------------------------------------------------------------------------
class ComplexAttribute(AttributeTemplate):
    def resolve_real(self, ty):
        return self.innertype

    def resolve_imag(self, ty):
        return self.innertype


@builtin_attr
class Complex64Attribute(ComplexAttribute):
    key = types.complex64
    innertype = types.float32


@builtin_attr
class Complex128Attribute(ComplexAttribute):
    key = types.complex128
    innertype = types.float64

#-------------------------------------------------------------------------------

@builtin_attr
class NumbaTypesModuleAttribute(AttributeTemplate):
    key = types.Module(types)

    def resolve_int8(self, mod):
        return types.Function(ToInt8)

    def resolve_int16(self, mod):
        return types.Function(ToInt16)

    def resolve_int32(self, mod):
        return types.Function(ToInt32)

    def resolve_int64(self, mod):
        return types.Function(ToInt64)

    def resolve_uint8(self, mod):
        return types.Function(ToUint8)

    def resolve_uint16(self, mod):
        return types.Function(ToUint16)

    def resolve_uint32(self, mod):
        return types.Function(ToUint32)

    def resolve_uint64(self, mod):
        return types.Function(ToUint64)

    def resolve_float32(self, mod):
        return types.Function(ToFloat32)

    def resolve_float64(self, mod):
        return types.Function(ToFloat64)

    def resolve_complex64(self, mod):
        return types.Function(ToComplex64)

    def resolve_complex128(self, mod):
        return types.Function(ToComplex128)


class Caster(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        [a] = args
        if a in types.number_domain:
            return signature(self.key, a)


class ToInt8(Caster):
    key = types.int8


class ToInt16(Caster):
    key = types.int16


class ToInt32(Caster):
    key = types.int32


class ToInt64(Caster):
    key = types.int64


class ToUint8(Caster):
    key = types.uint8


class ToUint16(Caster):
    key = types.uint16


class ToUint32(Caster):
    key = types.uint32


class ToUint64(Caster):
    key = types.uint64


class ToFloat32(Caster):
    key = types.float32


class ToFloat64(Caster):
    key = types.float64


class ToComplex64(Caster):
    key = types.complex64


class ToComplex128(Caster):
    key = types.complex128


builtin_global(types, types.Module(types))
builtin_global(types.int8, types.Function(ToInt8))
builtin_global(types.int16, types.Function(ToInt16))
builtin_global(types.int32, types.Function(ToInt32))
builtin_global(types.int64, types.Function(ToInt64))
builtin_global(types.uint8, types.Function(ToUint8))
builtin_global(types.uint16, types.Function(ToUint16))
builtin_global(types.uint32, types.Function(ToUint32))
builtin_global(types.uint64, types.Function(ToUint64))
builtin_global(types.float32, types.Function(ToFloat32))
builtin_global(types.float64, types.Function(ToFloat64))
builtin_global(types.complex64, types.Function(ToComplex64))
builtin_global(types.complex128, types.Function(ToComplex128))

#------------------------------------------------------------------------------


class Max(AbstractTemplate):
    key = max

    def generic(self, args, kws):
        assert not kws

        for a in args:
            if a not in types.number_domain:
                raise TypeError("max() only support for numbers")

        retty = self.context.unify_types(*args)
        return signature(retty, *args)


class Min(AbstractTemplate):
    key = min

    def generic(self, args, kws):
        assert not kws

        for a in args:
            if a not in types.number_domain:
                raise TypeError("min() only support for numbers")

        retty = self.context.unify_types(*args)
        return signature(retty, *args)


class Round(ConcreteTemplate):
    key = round
    cases = [
        signature(types.float32, types.float32),
        signature(types.float64, types.float64),
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

        if arg not in types.number_domain:
            raise TypeError("bool() only support for numbers")

        return signature(types.boolean, arg)


class Int(AbstractTemplate):
    key = int

    def generic(self, args, kws):
        assert not kws

        [arg] = args

        if arg not in types.number_domain:
            raise TypeError("int() only support for numbers")

        if arg in types.complex_domain:
            raise TypeError("int() does not support complex")

        if arg in types.integer_domain:
            return signature(arg, arg)

        if arg in types.real_domain:
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
            return signature(types.complex128, arg)

        elif len(args) == 2:
            [real, imag] = args

            if (real not in types.number_domain or
                        imag not in types.number_domain):
                raise TypeError("complex() only support for numbers")
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
        [it] = args
        if isinstance(it, types.IterableType):
            enumerate_type = types.EnumerateType(it)
            return signature(enumerate_type, it)


builtin_global(enumerate, types.Function(Enumerate))
