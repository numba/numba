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
    cases = [signature(ty, ty) for ty in types.signed_domain]


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
        ary, idx = args
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
