from __future__ import print_function, division, absolute_import


from numba import types, intrinsics
from numba.utils import PYVERSION, RANGE_ITER_OBJECTS
from numba.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                    AbstractTemplate, builtin_global, builtin,
                                    builtin_attr, signature, bound_function)

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
    cases = [signature(types.intp, types.intp, types.intp)]
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


class TupleCompare(AbstractTemplate):
    def generic(self, args, kws):
        [lhs, rhs] = args
        tuple_types = (types.Tuple, types.UniTuple)
        if isinstance(lhs, tuple_types) and isinstance(rhs, tuple_types):
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


def normalize_index(index):
    if isinstance(index, types.UniTuple):
        if index.dtype in types.integer_domain:
            idxtype = types.intp if index.dtype.signed else types.uintp
            return types.UniTuple(idxtype, len(index))
        elif index.dtype == types.slice3_type:
            return index

    elif isinstance(index, types.Tuple):
        for ty in index:
            if (ty not in types.integer_domain and ty != types.slice3_type):
                raise TypeError('Type %s of index %s is unsupported for indexing'
                                 % (ty, index))
        return index

    elif index == types.slice3_type:
        return types.slice3_type

    elif isinstance(index, types.Integer):
        return types.intp if index.signed else types.uintp


@builtin
class GetItemUniTuple(AbstractTemplate):
    key = "getitem"

    def generic(self, args, kws):
        tup, idx = args
        if isinstance(tup, types.UniTuple):
            return signature(tup.dtype, tup, normalize_index(idx))


@builtin
class GetItemBuffer(AbstractTemplate):
    key = "getitem"

    def generic(self, args, kws):
        assert not kws
        [ary, idx] = args
        if not isinstance(ary, types.Buffer):
            return

        idx = normalize_index(idx)
        if idx is None:
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
        elif isinstance(idx, types.Integer):
            if ary.ndim == 1:
                res = ary.dtype
            elif not ary.slice_is_copy and ary.ndim > 1:
                res = ary.copy(ndim=ary.ndim - 1)
            else:
                return

        else:
            raise Exception("unreachable: index type of %s" % idx)

        if isinstance(res, types.Buffer) and res.slice_is_copy:
            # Avoid view semantics when the original type creates a copy
            # when slicing.
            return

        return signature(res, ary, idx)


@builtin
class GetItemCPointer(AbstractTemplate):
    key = "getitem"

    def generic(self, args, kws):
        assert not kws
        ptr, idx = args
        if isinstance(ptr, types.CPointer) and isinstance(idx, types.Integer):
            return signature(ptr.dtype, ptr, normalize_index(idx))


@builtin
class SetItemBuffer(AbstractTemplate):
    key = "setitem"

    def generic(self, args, kws):
        assert not kws
        ary, idx, val = args
        if isinstance(ary, types.Buffer):
            if not ary.mutable:
                raise TypeError("Immutable array")
            return signature(types.none, ary, normalize_index(idx), ary.dtype)


@builtin
class SetItemCPointer(AbstractTemplate):
    key = "setitem"

    def generic(self, args, kws):
        assert not kws
        ptr, idx, val = args
        if isinstance(ptr, types.CPointer) and isinstance(idx, types.Integer):
            return signature(types.none, ptr, normalize_index(idx), ptr.dtype)


@builtin
class Len(AbstractTemplate):
    key = types.len_type

    def generic(self, args, kws):
        assert not kws
        (val,) = args
        if isinstance(val, (types.Buffer, types.Tuple, types.UniTuple)):
            return signature(types.intp, val)


#-------------------------------------------------------------------------------

@builtin_attr
class ArrayAttribute(AttributeTemplate):
    key = types.Array

    def resolve_itemsize(self, ary):
        return types.intp

    def resolve_shape(self, ary):
        return types.UniTuple(types.intp, ary.ndim)

    def resolve_strides(self, ary):
        return types.UniTuple(types.intp, ary.ndim)

    def resolve_ndim(self, ary):
        return types.intp

    def resolve_size(self, ary):
        return types.intp

    def resolve_flat(self, ary):
        return types.NumpyFlatType(ary)

    def resolve_ctypes(self, ary):
        return types.ArrayCTypes(ary)

    def generic_resolve(self, ary, attr):
        if isinstance(ary.dtype, types.Record):
            if attr in ary.dtype.fields:
                return types.Array(ary.dtype.typeof(attr), ndim=ary.ndim,
                                   layout='A')


@builtin_attr
class ArrayCTypesAttribute(AttributeTemplate):
    key = types.ArrayCTypes

    def resolve_data(self, ctinfo):
        return types.uintp



@builtin_attr
class NestedArrayAttribute(ArrayAttribute):
    key = types.NestedArray


def _expand_integer(ty):
    """
    If *ty* is an integer, expand it to a machine int (like Numpy).
    """
    if isinstance(ty, types.Integer):
        if ty.signed:
            return max(types.intp, ty)
        else:
            return max(types.uintp, ty)
    else:
        return ty

def generic_homog(self, args, kws):
    assert not args
    assert not kws
    return signature(self.this.dtype, recvr=self.this)

def generic_expand(self, args, kws):
    return signature(_expand_integer(self.this.dtype), recvr=self.this)

def generic_expand_cumulative(self, args, kws):
    assert isinstance(self.this, types.Array)
    return_type = types.Array(dtype=_expand_integer(self.this.dtype),
                              ndim=1, layout='C')
    return signature(return_type, recvr=self.this)

def generic_hetero_real(self, args, kws):
    assert not args
    assert not kws
    if self.this.dtype in types.integer_domain:
        return signature(types.float64, recvr=self.this)
    return signature(self.this.dtype, recvr=self.this)

def generic_index(self, args, kws):
    assert not args
    assert not kws
    return signature(types.intp, recvr=self.this)

def install_array_method(name, generic):
    my_attr = {"key": "array." + name, "generic": generic}
    temp_class = type("Array_" + name, (AbstractTemplate,), my_attr)

    def array_attribute_attachment(self, ary):
        return types.BoundFunction(temp_class, ary)

    setattr(ArrayAttribute, "resolve_" + name, array_attribute_attachment)

# Functions that return the same type as the array
for fname in ["min", "max"]:
    install_array_method(fname, generic_homog)

# Functions that return a machine-width type, to avoid overflows
for fname in ["sum", "prod"]:
    install_array_method(fname, generic_expand)

# Functions that return a machine-width type, to avoid overflows
for fname in ["cumsum", "cumprod"]:
    install_array_method(fname, generic_expand_cumulative)

# Functions that require integer arrays get promoted to float64 return
for fName in ["mean", "var", "std"]:
    install_array_method(fName, generic_hetero_real)

# Functions that return an index (intp)
install_array_method("argmin", generic_index)
install_array_method("argmax", generic_index)


@builtin
class CmpOpEqArray(AbstractTemplate):
    key = '=='

    def generic(self, args, kws):
        assert not kws
        [va, vb] = args
        if isinstance(va, types.Array) and va == vb:
            return signature(va.copy(dtype=types.boolean), va, vb)


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
class ComplexAttribute(AttributeTemplate):

    def resolve_real(self, ty):
        return self.innertype

    def resolve_imag(self, ty):
        return self.innertype

    @bound_function("complex.conjugate")
    def resolve_conjugate(self, ty, args, kws):
        assert not args
        assert not kws
        return signature(ty)


def register_complex_attributes(ty):
    @builtin_attr
    class ConcreteComplexAttribute(ComplexAttribute):
        key = ty
        try:
            innertype = ty.underlying_float
        except AttributeError:
            innertype = ty

for ty in types.number_domain:
    register_complex_attributes(ty)

#-------------------------------------------------------------------------------

def register_casters(register_global):
    nb_types = set(types.number_domain)
    nb_types.add(types.bool_)

    for restype in nb_types:
        class Caster(AbstractTemplate):
            key = restype

            def generic(self, args, kws):
                assert not kws
                [a] = args
                if a in nb_types:
                    return signature(self.key, a)

        register_global(restype, types.NumberClass(restype, Caster))

register_casters(builtin_global)

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
            signature(types.int32, types.float32),
            signature(types.int64, types.float64),
        ]
    cases += [
        signature(types.float32, types.float32, types.int32),
        signature(types.float64, types.float64, types.int32),
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
