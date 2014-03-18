from __future__ import print_function, division, absolute_import
import itertools
from numba import typing, types
from numba.typing.templates import (AttributeTemplate, ConcreteTemplate,
				    signature)

FunctionContext = types.OpaqueType('class.impala_udf::FunctionContext')


class ImpalaValue(types.Type):
    pass

AnyVal = ImpalaValue('AnyVal')

BooleanVal = ImpalaValue('BooleanVal')
BooleanValType = types.Dummy('BooleanValType')


class BooleanValCtor(ConcreteTemplate):
    key = BooleanValType
    cases = [signature(BooleanVal, types.int8)]


class BooleanValValueAttr(AttributeTemplate):
    key = BooleanVal

    def resolve_is_null(self, val):
	"""
	BooleanVal::is_null
	"""
	return types.boolean

    def resolve_val(self, val):
	"""
	BooleanVal::val
	"""
	return types.int8


class BooleanValTypeAttr(AttributeTemplate):
    key = BooleanValType

    def resolve_null(self, typ):
	"""
	BooleanVal::null
	"""
	return BooleanVal


TinyIntVal = ImpalaValue('TinyIntVal')
TinyIntValType = types.Dummy('TinyIntValType')


class TinyIntValCtor(ConcreteTemplate):
    key = TinyIntValType
    cases = [signature(TinyIntVal, types.int8)]


class TinyIntValValueAttr(AttributeTemplate):
    key = TinyIntVal

    def resolve_is_null(self, val):
	"""
	TinyIntVal::is_null
	"""
	return types.boolean

    def resolve_val(self, val):
	"""
	TinyIntVal::val
	"""
	return types.int8


class TinyIntValTypeAttr(AttributeTemplate):
    key = TinyIntValType

    def resolve_null(self, typ):
	"""
	TinyIntVal::null
	"""
	return TinyIntVal

SmallIntVal = ImpalaValue('SmallIntVal')
SmallIntValType = types.Dummy('SmallIntValType')


class SmallIntValCtor(ConcreteTemplate):
    key = SmallIntValType
    cases = [signature(SmallIntVal, types.int16)]


class SmallIntValValueAttr(AttributeTemplate):
    key = SmallIntVal

    def resolve_is_null(self, val):
	"""
	SmallIntVal::is_null
	"""
	return types.boolean

    def resolve_val(self, val):
	"""
	SmallIntVal::val
	"""
	return types.int16


class SmallIntValTypeAttr(AttributeTemplate):
    key = SmallIntValType

    def resolve_null(self, typ):
	"""
	SmallIntVal::null
	"""
	return SmallIntVal


IntVal = ImpalaValue('IntVal')
IntValType = types.Dummy('IntValType')


class IntValCtor(ConcreteTemplate):
    key = IntValType
    cases = [signature(IntVal, types.int32)]


class IntValValueAttr(AttributeTemplate):
    key = IntVal

    def resolve_is_null(self, val):
	"""
	IntVal::is_null
	"""
	return types.boolean

    def resolve_val(self, val):
	"""
	IntVal::val
	"""
	return types.int32


class IntValTypeAttr(AttributeTemplate):
    key = IntValType

    def resolve_null(self, typ):
	"""
	IntVal::null
	"""
	return IntVal


BigIntVal = ImpalaValue('BigIntVal')
BigIntValType = types.Dummy('BigIntValType')


class BigIntValCtor(ConcreteTemplate):
    key = BigIntValType
    cases = [signature(BigIntVal, types.int64)]


class BigIntValValueAttr(AttributeTemplate):
    key = BigIntVal

    def resolve_is_null(self, val):
	"""
	BigIntVal::is_null
	"""
	return types.boolean

    def resolve_val(self, val):
	"""
	BigIntVal::val
	"""
	return types.int64


class BigIntValTypeAttr(AttributeTemplate):
    key = BigIntValType

    def resolve_null(self, typ):
	"""
	BigIntVal::null
	"""
	return BigIntVal


FloatVal = ImpalaValue('FloatVal')
FloatValType = types.Dummy('FloatValType')


class FloatValCtor(ConcreteTemplate):
    key = FloatValType
    cases = [signature(FloatVal, types.float32)]


class FloatValValueAttr(AttributeTemplate):
    key = FloatVal

    def resolve_is_null(self, val):
	"""
	FloatVal::is_null
	"""
	return types.boolean

    def resolve_val(self, val):
	"""
	FloatVal::val
	"""
	return types.float32


class FloatValTypeAttr(AttributeTemplate):
    key = FloatValType

    def resolve_null(self, typ):
	"""
	FloatVal::null
	"""
	return FloatVal


DoubleVal = ImpalaValue('DoubleVal')
DoubleValType = types.Dummy('DoubleValType')


class DoubleValCtor(ConcreteTemplate):
    key = DoubleValType
    cases = [signature(DoubleVal, types.float64)]


class DoubleValValueAttr(AttributeTemplate):
    key = DoubleVal

    def resolve_is_null(self, val):
	"""
	DoubleVal::is_null
	"""
	return types.boolean

    def resolve_val(self, val):
	"""
	DoubleVal::val
	"""
	return types.float64


class DoubleValTypeAttr(AttributeTemplate):
    key = DoubleValType

    def resolve_null(self, typ):
	"""
	DoubleVal::null
	"""
	return DoubleVal


StringVal = ImpalaValue('StringVal')
StringValType = types.Dummy('StringValType')


class StringValCtor(ConcreteTemplate):
    key = StringValType
    cases = [signature(StringVal, types.CPointer(types.uint8), types.int32)]


class StringValValueAttr(AttributeTemplate):
    key = StringVal

    def resolve_is_null(self, val):
	"""
	StringVal::is_null
	"""
	return types.boolean

    def resolve_len(self, val):
	"""
	StringVal::len
	"""
	return types.int32

    def resolve_ptr(self, val):
	"""
	StringVal::ptr
	"""
	return types.CPointer(types.uint8)


class StringValTypeAttr(AttributeTemplate):
    key = StringValType

    def resolve_null(self, typ):
	"""
	StringVal::null
	"""
	return StringVal


class LenStringVal(ConcreteTemplate):
    key = types.len_type
    cases = [signature(types.int32, StringVal)]


class GetItemStringVal(ConcreteTemplate):
    key = "getitem"
    cases = [signature(types.uint8, StringVal, types.int64)]


class EqPointerUint8(ConcreteTemplate):
    key = '=='
    cases = [signature(types.boolean, types.CPointer(types.uint8), types.CPointer(types.uint8))]


class EqStringVal(ConcreteTemplate):
    key = '=='
    cases = [signature(types.boolean, StringVal, StringVal)]


class StringValToInt16(ConcreteTemplate):
    key = 'StringValToInt16'
    cases = [signature(types.int16, StringVal)]


class BinOpIs(ConcreteTemplate):
    key = 'is'
    cases = [signature(types.int8, AnyVal, types.none)]

def _register_impala_numeric_type_conversions(base):
    impala_integral = (BooleanVal, TinyIntVal, SmallIntVal, IntVal, BigIntVal)
    impala_float = (FloatVal, DoubleVal)
    impala_all = impala_integral + impala_float
    numba_integral = (types.boolean, types.int8, types.int16, types.int32, types.int64)
    numba_float = (types.float32, types.float64)
    numba_all = numba_integral + numba_float
    all_numeric = impala_all + numba_all

    # first, all Impala numeric types can cast to all others
    for a, b in itertools.product(impala_all, all_numeric):
	base.tm.set_unsafe_convert(a, b)
	base.tm.set_unsafe_convert(b, a)

    # match Numba-Impala types
    for a, b in zip(impala_all, numba_all):
	# base.tm.set_safe_convert(a, b)
	# base.tm.set_safe_convert(b, a)
	base.tm.set_unsafe_convert(a, b)
	base.tm.set_promote(b, a)

    # set up promotions
    for i in range(len(impala_integral)):
	for j in range(i + 1, len(numba_integral)):
	    base.tm.set_promote(impala_integral[i], numba_integral[j])
	    base.tm.set_promote(numba_integral[i], impala_integral[j])
	    base.tm.set_promote(impala_integral[i], impala_integral[j])
    for i in range(len(impala_float)):
	for j in range(i + 1, len(numba_float)):
	    base.tm.set_promote(impala_float[i], numba_float[j])
	    base.tm.set_promote(numba_float[i], impala_float[j])
	    base.tm.set_promote(impala_float[i], impala_float[j])

    # boolean safely promotes to everything
    for b in impala_all:
	base.tm.set_promote(types.boolean, b)
    for b in all_numeric:
	base.tm.set_promote(BooleanVal, b)

    # int to float conversions
    for a in impala_integral[:-2]:
	base.tm.set_safe_convert(a, types.float32)
	base.tm.set_safe_convert(a, types.float64)
	base.tm.set_safe_convert(a, FloatVal)
	base.tm.set_safe_convert(a, DoubleVal)
    for a in numba_integral[:-2]:
	base.tm.set_safe_convert(a, FloatVal)
	base.tm.set_safe_convert(a, DoubleVal)
    base.tm.set_safe_convert(impala_integral[-2], types.float64)
    base.tm.set_safe_convert(impala_integral[-2], DoubleVal)
    base.tm.set_safe_convert(numba_integral[-2], DoubleVal)

    # *Val to AnyVal
    for a in impala_all:
	base.tm.set_unsafe_convert(a, AnyVal)

    for a in impala_all:
	base.tm.set_safe_convert(types.none, a)

def _register_impala_string_type_conversions(base):
    base.tm.set_unsafe_convert(types.CPointer(types.uint8), types.Dummy('void*'))


def impala_typing_context():
    base = typing.Context()

    _register_impala_numeric_type_conversions(base)
    _register_impala_string_type_conversions(base)

    base.insert_function(BinOpIs(base))

    base.insert_global(BooleanVal, BooleanValType)
    base.insert_function(BooleanValCtor(base))
    base.insert_attributes(BooleanValValueAttr(base))
    base.insert_attributes(BooleanValTypeAttr(base))

    base.insert_global(TinyIntVal, TinyIntValType)
    base.insert_function(TinyIntValCtor(base))
    base.insert_attributes(TinyIntValValueAttr(base))
    base.insert_attributes(TinyIntValTypeAttr(base))

    base.insert_global(SmallIntVal, SmallIntValType)
    base.insert_function(SmallIntValCtor(base))
    base.insert_attributes(SmallIntValValueAttr(base))
    base.insert_attributes(SmallIntValTypeAttr(base))

    base.insert_global(IntVal, IntValType)
    base.insert_function(IntValCtor(base))
    base.insert_attributes(IntValValueAttr(base))
    base.insert_attributes(IntValTypeAttr(base))

    base.insert_global(BigIntVal, BigIntValType)
    base.insert_function(BigIntValCtor(base))
    base.insert_attributes(BigIntValValueAttr(base))
    base.insert_attributes(BigIntValTypeAttr(base))

    base.insert_global(FloatVal, FloatValType)
    base.insert_function(FloatValCtor(base))
    base.insert_attributes(FloatValValueAttr(base))
    base.insert_attributes(FloatValTypeAttr(base))

    base.insert_global(DoubleVal, DoubleValType)
    base.insert_function(DoubleValCtor(base))
    base.insert_attributes(DoubleValValueAttr(base))
    base.insert_attributes(DoubleValTypeAttr(base))

    base.insert_global(StringVal, StringValType)
    base.insert_function(StringValCtor(base))
    base.insert_attributes(StringValValueAttr(base))
    base.insert_attributes(StringValTypeAttr(base))
    base.insert_function(LenStringVal(base))
    base.insert_function(EqStringVal(base))
    base.insert_function(GetItemStringVal(base))
    base.insert_function(EqPointerUint8(base))
    base.insert_global(StringValToInt16, types.Function(StringValToInt16))

    return base
