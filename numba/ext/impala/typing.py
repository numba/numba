from __future__ import print_function, division, absolute_import
import itertools
from numba import typing, types
from numba.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                    signature)


_globals = []
_functions = []
_attributes = []

def register_global(gv, gty):
    _globals.append((gv, gty))

def register_function(func):
    _functions.append(func)
    return func

def register_attribute(attr):
    _attributes.append(attr)
    return attr


# Impala types

FunctionContext = types.OpaqueType('class.impala_udf::FunctionContext')

class ImpalaValue(types.Type):
    pass

AnyVal = ImpalaValue('AnyVal')

BooleanVal = ImpalaValue('BooleanVal')
BooleanValType = types.Dummy('BooleanValType')
register_global(BooleanVal, BooleanValType)

TinyIntVal = ImpalaValue('TinyIntVal')
TinyIntValType = types.Dummy('TinyIntValType')
register_global(TinyIntVal, TinyIntValType)

SmallIntVal = ImpalaValue('SmallIntVal')
SmallIntValType = types.Dummy('SmallIntValType')
register_global(SmallIntVal, SmallIntValType)

IntVal = ImpalaValue('IntVal')
IntValType = types.Dummy('IntValType')
register_global(IntVal, IntValType)

BigIntVal = ImpalaValue('BigIntVal')
BigIntValType = types.Dummy('BigIntValType')
register_global(BigIntVal, BigIntValType)

FloatVal = ImpalaValue('FloatVal')
FloatValType = types.Dummy('FloatValType')
register_global(FloatVal, FloatValType)

DoubleVal = ImpalaValue('DoubleVal')
DoubleValType = types.Dummy('DoubleValType')
register_global(DoubleVal, DoubleValType)

StringVal = ImpalaValue('StringVal')
StringValType = types.Dummy('StringValType')
register_global(StringVal, StringValType)


# *Val ctors

def _ctor_factory(Val, ValType, argty):
    class ValCtor(ConcreteTemplate):
        key = ValType
        cases = [signature(Val, argty)]
    return register_function(ValCtor)

BooleanValCtor = _ctor_factory(BooleanVal, BooleanValType, types.int8)
TinyIntValCtor = _ctor_factory(TinyIntVal, TinyIntValType, types.int8)
SmallIntValCtor = _ctor_factory(SmallIntVal, SmallIntValType, types.int16)
IntValCtor = _ctor_factory(IntVal, IntValType, types.int32)
BigIntValCtor = _ctor_factory(BigIntVal, BigIntValType, types.int64)
FloatValCtor = _ctor_factory(FloatVal, FloatValType, types.float32)
DoubleValCtor = _ctor_factory(DoubleVal, DoubleValType, types.float64)
StringValCtor = _ctor_factory(StringVal, StringValType, types.CPointer(types.char))


# *Val attributes

def _attr_factory(Val, ValType, retty):
    class ValAttr(AttributeTemplate):
        key = Val
        
        def resolve_is_null(self, val):
            """*Val::is_null"""
            return types.boolean
        
        def resolve_val(self, val):
            """*Val::val"""
            return retty
    return register_attribute(ValAttr)

BooleanValAttr = _attr_factory(BooleanVal, BooleanValType, types.int8)
TinyIntValAttr = _attr_factory(TinyIntVal, TinyIntValType, types.int8)
SmallIntValAttr = _attr_factory(SmallIntVal, SmallIntValType, types.int16)
IntValAttr = _attr_factory(IntVal, IntValType, types.int32)
BigIntValAttr = _attr_factory(BigIntVal, BigIntValType, types.int64)
FloatValAttr = _attr_factory(FloatVal, FloatValType, types.float32)
DoubleValAttr = _attr_factory(DoubleVal, DoubleValType, types.float64)

@register_attribute
class StringValAttr(AttributeTemplate):
    key = StringVal

    def resolve_is_null(self, val):
        """StringVal::is_null"""
        return types.boolean

    def resolve_len(self, val):
        """StringVal::len"""
        return types.int32

    def resolve_ptr(self, val):
        """StringVal::ptr"""
        return types.CPointer(types.uint8)


# register "builtins"

@register_function
class LenStringVal(ConcreteTemplate):
    key = types.len_type
    cases = [signature(types.int32, StringVal)]


@register_function
class CmpOpEqPtr(ConcreteTemplate):
    key = '=='
    cases = [signature(types.boolean, types.CPointer(types.uint8), types.CPointer(types.uint8))]


@register_function
class CmpOpEqStringVal(ConcreteTemplate):
    key = '=='
    cases = [signature(types.boolean, StringVal, StringVal)]


@register_function
class BinOpIs(ConcreteTemplate):
    key = 'is'
    cases = [signature(types.int8, AnyVal, types.none)]


@register_function
class GetItemStringVal(ConcreteTemplate):
    key = "getitem"
    cases = [signature(types.uint8, StringVal, types.int64)]


# type conversions

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

    # *Val to AnyVal (numeric)
    for a in impala_all:
        base.tm.set_unsafe_convert(a, AnyVal)

    for a in impala_all:
        base.tm.set_safe_convert(types.none, a)

def _register_impala_other_type_conversions(base):
    # base.tm.set_unsafe_convert(types.CPointer(types.uint8), types.Dummy('void*'))
    base.tm.set_safe_convert(types.string, StringVal)
    base.tm.set_unsafe_convert(StringVal, AnyVal)
    base.tm.set_safe_convert(types.none, StringVal)
    base.tm.set_safe_convert(types.none, AnyVal)


# the Impala typing context

def impala_typing_context():
    base = typing.Context()

    _register_impala_numeric_type_conversions(base)
    _register_impala_other_type_conversions(base)

    for (gv, gty) in _globals:
        base.insert_global(gv, gty)
    
    for func in _functions:
        base.insert_function(func(base))
    
    for attr in _attributes:
        base.insert_attributes(attr(base))

    return base
