from __future__ import print_function, division, absolute_import
from numba import types
from numba.utils import PYVERSION
from numba.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                    AbstractTemplate, builtin_global, builtin,
                                    signature)


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
class Abs(ConcreteTemplate):
    key = types.abs_type
    intcases = [signature(ty, ty) for ty in types.signed_domain]
    realcases = [signature(ty, ty) for ty in types.real_domain]
    cases = intcases + realcases


@builtin
class Slice(ConcreteTemplate):
    key = types.slice_type
    cases = [
        signature(types.slice2_type, types.intp, types.intp),
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
class GetIter(ConcreteTemplate):
    key = "getiter"
    cases = [
        signature(types.range_iter32_type, types.range_state32_type),
        signature(types.range_iter64_type, types.range_state64_type),
    ]


@builtin
class GetIterUniTuple(AbstractTemplate):
    key = "getiter"

    def generic(self, args, kws):
        assert not kws
        [tup] = args
        if isinstance(tup, types.UniTuple):
            return signature(types.UniTupleIter(tup), tup)


@builtin
class IterNext(ConcreteTemplate):
    key = "iternext"
    cases = [
        signature(types.int32, types.range_iter32_type),
        signature(types.int64, types.range_iter64_type),
    ]


@builtin
class IterNextSafe(AbstractTemplate):
    key = "iternextsafe"

    def generic(self, args, kws):
        assert not kws
        [tupiter] = args
        if isinstance(tupiter, types.UniTupleIter):
            return signature(tupiter.unituple.dtype, tupiter)


@builtin
class IterValid(ConcreteTemplate):
    key = "itervalid"
    cases = [
        signature(types.boolean, types.range_iter32_type),
        signature(types.boolean, types.range_iter64_type),
    ]


class BinOp(ConcreteTemplate):
    cases = [
        signature(types.uintp, types.uint8, types.uint8),
        signature(types.uintp, types.uint16, types.uint16),
        signature(types.uintp, types.uint32, types.uint32),
        signature(types.uint64, types.uint64, types.uint64),

        signature(types.intp, types.int8, types.int8),
        signature(types.intp, types.int16, types.int16),
        signature(types.intp, types.int32, types.int32),
        signature(types.int64, types.int64, types.int64),

        signature(types.float32, types.float32, types.float32),
        signature(types.float64, types.float64, types.float64),

        signature(types.complex64, types.complex64, types.complex64),
        signature(types.complex128, types.complex128, types.complex128),
    ]


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

    cases = [
        signature(types.uint8, types.uint8, types.uint8),
        signature(types.uint16, types.uint16, types.uint16),
        signature(types.uint32, types.uint32, types.uint32),
        signature(types.uint64, types.uint64, types.uint64),

        signature(types.int8, types.int8, types.int8),
        signature(types.int16, types.int16, types.int16),
        signature(types.int32, types.int32, types.int32),
        signature(types.int64, types.int64, types.int64),

        signature(types.float32, types.float32, types.float32),
        signature(types.float64, types.float64, types.float64),
    ]


@builtin
class BinOpTrueDiv(ConcreteTemplate):
    key = "/"

    cases = [
        signature(types.float64, types.uint8, types.uint8),
        signature(types.float64, types.uint16, types.uint16),
        signature(types.float64, types.uint32, types.uint32),
        signature(types.float64, types.uint64, types.uint64),

        signature(types.float64, types.int8, types.int8),
        signature(types.float64, types.int16, types.int16),
        signature(types.float64, types.int32, types.int32),
        signature(types.float64, types.int64, types.int64),


        signature(types.float32, types.float32, types.float32),
        signature(types.float64, types.float64, types.float64),

        signature(types.complex64, types.complex64, types.complex64),
        signature(types.complex128, types.complex128, types.complex128),

    ]

@builtin
class BinOpFloorDiv(ConcreteTemplate):
    key = "//"
    cases = [
        signature(types.int8, types.int8, types.int8),
        signature(types.int16, types.int16, types.int16),
        signature(types.int32, types.int32, types.int32),
        signature(types.int64, types.int64, types.int64),

        signature(types.uint8, types.uint8, types.uint8),
        signature(types.uint16, types.uint16, types.uint16),
        signature(types.uint32, types.uint32, types.uint32),
        signature(types.uint64, types.uint64, types.uint64),

        signature(types.int32, types.float32, types.float32),
        signature(types.int64, types.float64, types.float64),
    ]


@builtin
class BinOpPower(ConcreteTemplate):
    key = "**"
    cases = [
        signature(types.float64, types.float64, types.uint8),
        signature(types.float64, types.float64, types.uint16),
        signature(types.float64, types.float64, types.uint32),
        signature(types.float64, types.float64, types.uint64),

        signature(types.float64, types.float64, types.int8),
        signature(types.float64, types.float64, types.int16),
        signature(types.float64, types.float64, types.int32),
        signature(types.float64, types.float64, types.int64),
        signature(types.float32, types.float32, types.float32),
        signature(types.float64, types.float64, types.float64),

        signature(types.complex64, types.complex64, types.complex64),
        signature(types.complex128, types.complex128, types.complex128),
    ]


class BitwiseShiftOperation(ConcreteTemplate):
    cases = [
        signature(types.int8, types.int8, types.uint32),
        signature(types.int16, types.int16, types.uint32),
        signature(types.int32, types.int32, types.uint32),
        signature(types.int64, types.int64, types.uint32),

        signature(types.uint8, types.uint8, types.uint32),
        signature(types.uint16, types.uint16, types.uint32),
        signature(types.uint32, types.uint32, types.uint32),
        signature(types.uint64, types.uint64, types.uint32),
    ]


@builtin
class BitwiseLeftShift(BitwiseShiftOperation):
    key = "<<"


@builtin
class BitwiseRightShift(BitwiseShiftOperation):
    key = ">>"


class BitwiseLogicOperation(BinOp):
    cases = [
        signature(types.uintp, types.uint8, types.uint8),
        signature(types.uintp, types.uint16, types.uint16),
        signature(types.uintp, types.uint32, types.uint32),
        signature(types.uint64, types.uint64, types.uint64),

        signature(types.intp, types.int8, types.int8),
        signature(types.intp, types.int16, types.int16),
        signature(types.intp, types.int32, types.int32),
        signature(types.int64, types.int64, types.int64),
    ]


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

    cases = [
        signature(types.uint8, types.uint8),
        signature(types.uint16, types.uint16),
        signature(types.uint32, types.uint32),
        signature(types.uint64, types.uint64),

        signature(types.int8, types.int8),
        signature(types.int16, types.int16),
        signature(types.int32, types.int32),
        signature(types.int64, types.int64),
    ]


class UnaryOp(ConcreteTemplate):
    cases = [
        signature(types.uintp, types.uint8),
        signature(types.uintp, types.uint16),
        signature(types.uintp, types.uint32),
        signature(types.uint64, types.uint64),

        signature(types.intp, types.int8),
        signature(types.intp, types.int16),
        signature(types.intp, types.int32),
        signature(types.int64, types.int64),

        signature(types.float32, types.float32),
        signature(types.float64, types.float64),

        signature(types.complex64, types.complex64),
        signature(types.complex128, types.complex128),
    ]


@builtin
class UnaryNot(UnaryOp):
    key = "not"
    cases = [
        signature(types.boolean, types.uint8),
        signature(types.boolean, types.uint16),
        signature(types.boolean, types.uint32),
        signature(types.boolean, types.uint64),

        signature(types.boolean, types.int8),
        signature(types.boolean, types.int16),
        signature(types.boolean, types.int32),
        signature(types.boolean, types.int64),

        signature(types.boolean, types.float32),
        signature(types.boolean, types.float64),

        signature(types.boolean, types.complex64),
        signature(types.boolean, types.complex128),
    ]


@builtin
class UnaryNegate(UnaryOp):
    key = "-"
    cases = [
        signature(types.uintp, types.uint8),
        signature(types.uintp, types.uint16),
        signature(types.uintp, types.uint32),
        signature(types.uint64, types.uint64),

        signature(types.intp, types.int8),
        signature(types.intp, types.int16),
        signature(types.intp, types.int32),
        signature(types.int64, types.int64),

        signature(types.float32, types.float32),
        signature(types.float64, types.float64),

        signature(types.complex64, types.complex64),
        signature(types.complex128, types.complex128),
    ]


@builtin
class UnaryInvert(ConcreteTemplate):
    key = "~"

    cases = [
        signature(types.uintp, types.uint8),
        signature(types.uintp, types.uint16),
        signature(types.uintp, types.uint32),
        signature(types.uint64, types.uint64),

        signature(types.intp, types.int8),
        signature(types.intp, types.int16),
        signature(types.intp, types.int32),
        signature(types.int64, types.int64),
    ]


class CmpOp(ConcreteTemplate):
    cases = [
        signature(types.boolean, types.uint8, types.uint8),
        signature(types.boolean, types.uint16, types.uint16),
        signature(types.boolean, types.uint32, types.uint32),
        signature(types.boolean, types.uint64, types.uint64),

        signature(types.boolean, types.int8, types.int8),
        signature(types.boolean, types.int16, types.int16),
        signature(types.boolean, types.int32, types.int32),
        signature(types.boolean, types.int64, types.int64),

        signature(types.boolean, types.float32, types.float32),
        signature(types.boolean, types.float64, types.float64),
    ]


@builtin
class CmpOpLt(CmpOp):
    key = '<'


@builtin
class CmpOpLe(CmpOp):
    key = '<='


@builtin
class CmpOpGt(CmpOp):
    key = '>'


@builtin
class CmpOpGe(CmpOp):
    key = '>='


@builtin
class CmpOpEq(CmpOp):
    key = '=='


@builtin
class CmpOpNe(CmpOp):
    key = '!='


def normalize_index(index):
    if isinstance(index, types.UniTuple):
        return types.UniTuple(types.intp, index.count)

    elif isinstance(index, types.Tuple):
        for ty in index:
            if ty not in types.integer_domain:
                return
        return index

    elif index == types.slice3_type:
        return types.slice3_type

    elif index == types.slice2_type:
        return types.slice2_type

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
        if idx in (types.slice2_type, types.slice3_type):
            res = ary.copy(layout='A')
        elif isinstance(idx, types.UniTuple):
            if ary.ndim > len(idx):
                return
            elif ary.ndim < len(idx):
                return
            else:
                res = ary.dtype
        elif idx == types.intp:
            if ary.ndim != 1:
                return
            res = ary.dtype
        else:
            raise Exception("unreachable")

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

@builtin
class ArrayAttribute(AttributeTemplate):
    key = types.Array

    def resolve_shape(self, ary):
        return types.UniTuple(types.intp, ary.ndim)

    def resolve_strides(self, ary):
        return types.UniTuple(types.intp, ary.ndim)

    def resolve_ndim(self, ary):
        return types.intp

    def resolve_flatten(self, ary):
        return types.Method(Array_flatten, ary)


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


@builtin
class Complex64Attribute(ComplexAttribute):
    key = types.complex64
    innertype = types.float32


@builtin
class Complex128Attribute(ComplexAttribute):
    key = types.complex128
    innertype = types.float64

#-------------------------------------------------------------------------------

@builtin
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


builtin_global(max, types.Function(Max))
builtin_global(min, types.Function(Min))

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

