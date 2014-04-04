from __future__ import print_function, division, absolute_import
import llvm.core as lc
import llvm.passes as lp
import llvm.ee as le
from numba import types, cgutils, config, lowering
from numba.targets.base import BaseContext
from numba.targets.imputils import implement, impl_attribute
from .typing import (FunctionContext, AnyVal, BooleanVal, BooleanValType,
                     TinyIntVal, TinyIntValType, SmallIntVal, SmallIntValType,
                     IntVal, IntValType, BigIntVal, BigIntValType, FloatVal,
                     FloatValType, DoubleVal, DoubleValType, StringVal,
                     StringValType)


_functions = []
_attributes = []

def register_function(func):
    _functions.append(func)
    return func

def register_attribute(attr):
    _attributes.append(attr)
    return attr


# struct access utils

# these are necessary because cgutils.Structure assumes no nested types;
# the gep needs a (0, 0, 0) offset

def _get_is_null_pointer(builder, val):
    ptr = cgutils.inbound_gep(builder, val._getpointer(), 0, 0, 0)
    return ptr

def _get_is_null(builder, val):
    byte = builder.load(_get_is_null_pointer(builder, val))
    return builder.trunc(byte, lc.Type.int(1))

def _set_is_null(builder, val, is_null):
    byte = builder.zext(is_null, lc.Type.int(8))
    builder.store(byte, _get_is_null_pointer(builder, val))


# struct impls

class AnyValStruct(cgutils.Structure):
    _fields = [('is_null', types.boolean)]


class BooleanValStruct(cgutils.Structure):
    _fields = [('parent',  AnyVal),
               ('val',     types.int8),]


class TinyIntValStruct(cgutils.Structure):
    _fields = [('parent',  AnyVal),
               ('val',     types.int8),]


class SmallIntValStruct(cgutils.Structure):
    _fields = [('parent',  AnyVal),
               ('val',     types.int16),]


class IntValStruct(cgutils.Structure):
    _fields = [('parent',  AnyVal),
               ('val',     types.int32),]


class BigIntValStruct(cgutils.Structure):
    _fields = [('parent',  AnyVal),
               ('val',     types.int64),]


class FloatValStruct(cgutils.Structure):
    _fields = [('parent',  AnyVal),
               ('val',     types.float32),]


class DoubleValStruct(cgutils.Structure):
    _fields = [('parent',  AnyVal),
               ('val',     types.float64),]


class StringValStruct(cgutils.Structure):
    _fields = [('parent',  AnyVal),
               ('len',     types.int32),
               ('ptr',     types.CPointer(types.uint8))]


# ctor impls

def _ctor_factory(Struct, Type, *input_args):
    @implement(Type, *input_args)
    def Val_ctor(context, builder, sig, args):
        [x] = args
        v = Struct(context, builder)
        _set_is_null(builder, v, cgutils.false_bit)
        v.val = x
        return v._getvalue()
    return register_function(Val_ctor)

BooleanVal_ctor = _ctor_factory(BooleanValStruct, BooleanValType, types.int8)
TinyIntVal_ctor = _ctor_factory(TinyIntValStruct, TinyIntValType, types.int8)
SmallIntVal_ctor = _ctor_factory(SmallIntValStruct, SmallIntValType, types.int16)
IntVal_ctor = _ctor_factory(IntValStruct, IntValType, types.int32)
BigIntVal_ctor = _ctor_factory(BigIntValStruct, BigIntValType, types.int64)
FloatVal_ctor = _ctor_factory(FloatValStruct, FloatValType, types.float32)
DoubleVal_ctor = _ctor_factory(DoubleValStruct, DoubleValType, types.float64)

@register_function
@implement(StringValType, types.string)
def StringVal_ctor(context, builder, sig, args):
    """StringVal(types.string)"""
    [x] = args
    iv = StringValStruct(context, builder)
    _set_is_null(builder, iv, cgutils.false_bit)
    fndesc = lowering.describe_external('strlen', types.uintp, [types.CPointer(types.char)])
    func = context.declare_external_function(cgutils.get_module(builder), fndesc)
    strlen_x = context.call_external_function(builder, func, fndesc.argtypes, [x])
    len_x = builder.trunc(strlen_x, lc.Type.int(32))
    iv.len = len_x
    iv.ptr = x
    return iv._getvalue()




# *Val attributes

def _is_null_attr_factory(Struct, Val):
    @impl_attribute(Val, "is_null", types.boolean)
    def Val_is_null(context, builder, typ, value):
        v = Struct(context, builder, value=value)
        is_null = _get_is_null(builder, v)
        return is_null
    return register_attribute(Val_is_null)

def _val_attr_factory(Struct, Val, retty):
    @impl_attribute(Val, "val", retty)
    def Val_val(context, builder, typ, value):
        v = Struct(context, builder, value=value)
        return v.val
    return register_attribute(Val_val)

# *Val.is_null
BooleanVal_is_null = _is_null_attr_factory(BooleanValStruct, BooleanVal)
TinyIntVal_is_null = _is_null_attr_factory(TinyIntValStruct, TinyIntVal)
SmallIntVal_is_null = _is_null_attr_factory(SmallIntValStruct, SmallIntVal)
IntVal_is_null = _is_null_attr_factory(IntValStruct, IntVal)
BigIntVal_is_null = _is_null_attr_factory(BigIntValStruct, BigIntVal)
FloatVal_is_null = _is_null_attr_factory(FloatValStruct, FloatVal)
DoubleVal_is_null = _is_null_attr_factory(DoubleValStruct, DoubleVal)
StringVal_is_null = _is_null_attr_factory(StringValStruct, StringVal)

# *Val.val
BooleanVal_val = _val_attr_factory(BooleanValStruct, BooleanVal, types.int8)
TinyIntVal_val = _val_attr_factory(TinyIntValStruct, TinyIntVal, types.int8)
SmallIntVal_val = _val_attr_factory(SmallIntValStruct, SmallIntVal, types.int16)
IntVal_val = _val_attr_factory(IntValStruct, IntVal, types.int32)
BigIntVal_val = _val_attr_factory(BigIntValStruct, BigIntVal, types.int64)
FloatVal_val = _val_attr_factory(FloatValStruct, FloatVal, types.float32)
DoubleVal_val = _val_attr_factory(DoubleValStruct, DoubleVal, types.float64)

@register_attribute
@impl_attribute(StringVal, "len", types.int32)
def StringVal_len(context, builder, typ, value):
    """StringVal::len"""
    iv = StringValStruct(context, builder, value=value)
    return iv.len

@register_attribute
@impl_attribute(StringVal, "ptr", types.CPointer(types.uint8))
def StringVal_ptr(context, builder, typ, value):
    """StringVal::ptr"""
    iv = StringValStruct(context, builder, value=value)
    return iv.ptr


# impl "builtins"

@register_function
@implement('is', AnyVal, types.none)
def is_none_impl(context, builder, sig, args):
    [x, y] = args
    val = AnyValStruct(context, builder, value=x)
    return val.is_null

@register_function
@implement(types.len_type, StringVal)
def len_stringval_impl(context, builder, sig, args):
    [s] = args
    val = StringValStruct(context, builder, value=s)
    return val.len

@register_function
@implement("==", types.CPointer(types.uint8), types.CPointer(types.uint8))
def eq_ptr_impl(context, builder, sig, args):
    [p1, p2] = args
    return builder.icmp(lc.ICMP_EQ, p1, p2)

@register_function
@implement("==", StringVal, StringVal)
def eq_stringval(context, builder, sig, args):
    module = cgutils.get_module(builder)
    [s1, s2] = args
    sv1 = StringValStruct(context, builder, value=s1)
    sv2 = StringValStruct(context, builder, value=s2)
    # module.
    pass
    # TODO


TYPE_LAYOUT = {
    AnyVal: AnyValStruct,
    BooleanVal: BooleanValStruct,
    TinyIntVal: TinyIntValStruct,
    SmallIntVal: SmallIntValStruct,
    IntVal: IntValStruct,
    BigIntVal: BigIntValStruct,
    FloatVal: FloatValStruct,
    DoubleVal: DoubleValStruct,
    StringVal: StringValStruct,
}


class ImpalaTargetContext(BaseContext):
    _impala_types = (AnyVal, BooleanVal, TinyIntVal, SmallIntVal, IntVal,
                     BigIntVal, FloatVal, DoubleVal, StringVal)
    def init(self):
        self.tm = le.TargetMachine.new()
        # insert registered impls
        self.insert_func_defn(_functions)
        self.insert_attr_defn(_attributes)
        self.optimizer = self.build_pass_manager()

        # once per context
        self._fnctximpltype = lc.Type.opaque("FunctionContextImpl")
        fnctxbody = [lc.Type.pointer(self._fnctximpltype)]
        self._fnctxtype = lc.Type.struct(fnctxbody,
                                         name="class.impala_udf::FunctionContext")

    def cast(self, builder, val, fromty, toty):
        if config.DEBUG:
            print("CAST %s => %s" % (fromty, toty))

        if fromty not in self._impala_types and toty not in self._impala_types:
            return super(ImpalaTargetContext, self).cast(builder, val, fromty, toty)

        if fromty == toty:
            return val

        # handle NULLs and Nones
        if fromty == types.none and toty in self._impala_types:
            iv = TYPE_LAYOUT[toty](self, builder)
            _set_is_null(builder, iv, cgutils.true_bit)
            return iv._getvalue()
        if fromty in self._impala_types and toty == AnyVal:
            iv1 = TYPE_LAYOUT[fromty](self, builder, value=val)
            is_null = _get_is_null(builder, iv1)
            iv2 = AnyValStruct(self, builder)
            # this is equiv to _set_is_null, but changes the GEP bc of AnyVal's structure
            byte = builder.zext(is_null, lc.Type.int(8))
            builder.store(byte, cgutils.inbound_gep(builder, iv2._getpointer(), 0, 0))
            return iv2._getvalue()

        if fromty == BooleanVal:
            v = BooleanValStruct(self, builder, val)
            return self.cast(builder, v.val, types.boolean, toty)
        if fromty == TinyIntVal:
            v = TinyIntValStruct(self, builder, val)
            return self.cast(builder, v.val, types.int8, toty)
        if fromty == SmallIntVal:
            v = SmallIntValStruct(self, builder, val)
            return self.cast(builder, v.val, types.int16, toty)
        if fromty == IntVal:
            v = IntValStruct(self, builder, val)
            return self.cast(builder, v.val, types.int32, toty)
        if fromty == BigIntVal:
            v = BigIntValStruct(self, builder, val)
            return self.cast(builder, v.val, types.int64, toty)
        if fromty == FloatVal:
            v = FloatValStruct(self, builder, val)
            return self.cast(builder, v.val, types.float32, toty)
        if fromty == DoubleVal:
            v = DoubleValStruct(self, builder, val)
            return self.cast(builder, v.val, types.float64, toty)

        # no way fromty is a *Val starting here
        if toty == BooleanVal:
            val = super(ImpalaTargetContext, self).cast(builder, val, fromty, types.int8)
            return BooleanVal_ctor(self, builder, None, [val])
        if toty == TinyIntVal:
            val = super(ImpalaTargetContext, self).cast(builder, val, fromty, types.int8)
            return TinyIntVal_ctor(self, builder, None, [val])
        if toty == SmallIntVal:
            val = super(ImpalaTargetContext, self).cast(builder, val, fromty, types.int16)
            return SmallIntVal_ctor(self, builder, None, [val])
        if toty == IntVal:
            val = super(ImpalaTargetContext, self).cast(builder, val, fromty, types.int32)
            return IntVal_ctor(self, builder, None, [val])
        if toty == BigIntVal:
            val = super(ImpalaTargetContext, self).cast(builder, val, fromty, types.int64)
            return BigIntVal_ctor(self, builder, None, [val])
        if toty == FloatVal:
            val = super(ImpalaTargetContext, self).cast(builder, val, fromty, types.float32)
            return FloatVal_ctor(self, builder, None, [val])
        if toty == DoubleVal:
            val = super(ImpalaTargetContext, self).cast(builder, val, fromty, types.float64)
            return DoubleVal_ctor(self, builder, None, [val])
        if toty == StringVal:
            return stringval_ctor2(self, builder, None, [val])

        return super(ImpalaTargetContext, self).cast(builder, val, fromty, toty)

    def get_constant_string(self, builder, ty, val):
        assert ty == types.string
        literal = lc.Constant.stringz(val)
        gv = cgutils.get_module(builder).add_global_variable(literal.type, 'str_literal')
        gv.linkage = lc.LINKAGE_PRIVATE
        gv.initializer = literal
        gv.global_constant = True
        # gep gets pointer to first element of the constant byte array
        return gv.gep([lc.Constant.int(lc.Type.int(32), 0)] * 2)

    def get_constant_struct(self, builder, ty, val):
        # override for converting literals to *Vals, incl. None
        if ty in self._impala_types and val is None:
            iv = TYPE_LAYOUT[ty](self, builder)
            _set_is_null(builder, iv, cgutils.true_bit)
            return iv._getvalue()
        elif ty == BooleanVal:
            iv = BooleanValStruct(self, builder)
            _set_is_null(builder, iv, cgutils.false_bit)
            iv.val = lc.Constant.int(lc.Type.int(8), val)
            return iv._getvalue()
        elif ty == TinyIntVal:
            iv = TinyIntValStruct(self, builder)
            _set_is_null(builder, iv, cgutils.false_bit)
            iv.val = lc.Constant.int(lc.Type.int(8), val)
            return iv._getvalue()
        elif ty == SmallIntVal:
            iv = SmallIntValStruct(self, builder)
            _set_is_null(builder, iv, cgutils.false_bit)
            iv.val = lc.Constant.int(lc.Type.int(16), val)
            return iv._getvalue()
        elif ty == IntVal:
            iv = IntValStruct(self, builder)
            _set_is_null(builder, iv, cgutils.false_bit)
            iv.val = lc.Constant.int(lc.Type.int(32), val)
            return iv._getvalue()
        elif ty == BigIntVal:
            iv = BigIntValStruct(self, builder)
            _set_is_null(builder, iv, cgutils.false_bit)
            iv.val = lc.Constant.int(lc.Type.int(64), val)
            return iv._getvalue()
        elif ty == FloatVal:
            iv = FloatValStruct(self, builder)
            _set_is_null(builder, iv, cgutils.false_bit)
            iv.val = lc.Constant.real(lc.Type.float(), val)
            return iv._getvalue()
        elif ty == DoubleVal:
            iv = DoubleValStruct(self, builder)
            _set_is_null(builder, iv, cgutils.false_bit)
            iv.val = lc.Constant.real(lc.Type.double(), val)
            return iv._getvalue()
        elif ty == StringVal:
            iv = StringValStruct(self, builder)
            _set_is_null(builder, iv, cgutils.false_bit)
            iv.len = lc.Constant.int(lc.Type.int(32), len(val))
            iv.ptr = self.get_constant_string(builder, types.string, val)
            return iv._getvalue()
        else:
            return super(ImpalaTargetContext, self).get_constant_struct(builder, ty, val)

    def get_data_type(self, ty):
        if ty in TYPE_LAYOUT:
            return self.get_struct_type(TYPE_LAYOUT[ty])
        elif ty == FunctionContext:
            return lc.Type.pointer(self._fnctxtype)
        else:
            return super(ImpalaTargetContext, self).get_data_type(ty)

    def build_pass_manager(self):
        pms = lp.build_pass_managers(tm=self.tm, opt=3, loop_vectorize=True,
                                     fpm=False)
        return pms.pm

    def finalize(self, func, restype, argtypes):
        func.verify()
        func.linkage = lc.LINKAGE_INTERNAL

        module = func.module
        # Generate wrapper to adapt into Impala ABI
        abi = ABIHandling(self, func, restype, argtypes)
        wrapper = abi.build_wrapper("numba_udf." + func.name)
        module.verify()

        self.optimizer.run(module)
        return wrapper


class ABIHandling(object):
    """
    Adapt to C++ ABI for x86-64
    """
    def __init__(self, context, func, restype, argtypes):
        self.context = context
        self.func = func
        self.restype = restype
        self.argtypes = argtypes

    def build_wrapper(self, wrappername):
        abi_restype = self.get_abi_return_type(self.restype).pointee # should always ret pointer type
        abi_argtypes = [self.get_abi_argument_type(a)
                        for a in self.argtypes]
        fnty = lc.Type.function(abi_restype, abi_argtypes)
        wrapper = self.func.module.add_function(fnty, name=wrappername)

        builder = lc.Builder.new(wrapper.append_basic_block(''))
        status, res = self.context.call_function(builder, self.func, self.restype,
                                                 self.argtypes, wrapper.args)
        # FIXME ignoring error in function for now
        cres = self.convert_abi_return(builder, self.restype, res)
        builder.ret(cres)
        return wrapper

    def convert_abi_return(self, builder, ty, val):
        """
        Convert value to fit ABI requirement
        """
        if ty == BooleanVal:
            # Pack structure into int16
            # Endian specific
            iv = BooleanValStruct(self.context, builder, value=val)
            lower = builder.zext(_get_is_null(builder, iv), lc.Type.int(16))
            upper = builder.zext(iv.val, lc.Type.int(16))
            asint16 = builder.shl(upper, lc.Constant.int(lc.Type.int(16), 8))
            asint16 = builder.or_(asint16, lower)
            return asint16
        elif ty == TinyIntVal:
            # Pack structure into int16
            # Endian specific
            iv = TinyIntValStruct(self.context, builder, value=val)
            lower = builder.zext(_get_is_null(builder, iv), lc.Type.int(16))
            upper = builder.zext(iv.val, lc.Type.int(16))
            asint16 = builder.shl(upper, lc.Constant.int(lc.Type.int(16), 8))
            asint16 = builder.or_(asint16, lower)
            return asint16
        elif ty == SmallIntVal:
            # Pack structure into int32
            # Endian specific
            iv = SmallIntValStruct(self.context, builder, value=val)
            lower = builder.zext(_get_is_null(builder, iv), lc.Type.int(32))
            upper = builder.zext(iv.val, lc.Type.int(32))
            asint32 = builder.shl(upper, lc.Constant.int(lc.Type.int(32), 16))
            asint32 = builder.or_(asint32, lower)
            return asint32
        elif ty == IntVal:
            # Pack structure into int64
            # Endian specific
            iv = IntValStruct(self.context, builder, value=val)
            lower = builder.zext(_get_is_null(builder, iv), lc.Type.int(64))
            upper = builder.zext(iv.val, lc.Type.int(64))
            asint64 = builder.shl(upper, lc.Constant.int(lc.Type.int(64), 32))
            asint64 = builder.or_(asint64, lower)
            return asint64
        elif ty == BigIntVal:
            # Pack structure into { int8, int64 }
            # Endian specific
            iv = BigIntValStruct(self.context, builder, value=val)
            is_null = builder.zext(_get_is_null(builder, iv), lc.Type.int(8))
            asstructi8i64 = builder.insert_value(lc.Constant.undef(lc.Type.struct([lc.Type.int(8), lc.Type.int(64)])),
                                                 is_null,
                                                 0)
            asstructi8i64 = builder.insert_value(asstructi8i64, iv.val, 1)
            return asstructi8i64
        elif ty == FloatVal:
            # Pack structure into int64
            # Endian specific
            iv = FloatValStruct(self.context, builder, value=val)
            lower = builder.zext(_get_is_null(builder, iv), lc.Type.int(64))
            asint32 = builder.bitcast(iv.val, lc.Type.int(32))
            upper = builder.zext(asint32, lc.Type.int(64))
            asint64 = builder.shl(upper, lc.Constant.int(lc.Type.int(64), 32))
            asint64 = builder.or_(asint64, lower)
            return asint64
        elif ty == DoubleVal:
            # Pack structure into { int8, double }
            # Endian specific
            iv = DoubleValStruct(self.context, builder, value=val)
            is_null = builder.zext(_get_is_null(builder, iv), lc.Type.int(8))
            asstructi8double = builder.insert_value(lc.Constant.undef(lc.Type.struct([lc.Type.int(8), lc.Type.double()])),
                                                    is_null,
                                                    0)
            asstructi8double = builder.insert_value(asstructi8double, iv.val, 1)
            return asstructi8double
        elif ty == StringVal:
            # Pack structure into { int64, int8* }
            # Endian specific
            iv = StringValStruct(self.context, builder, value=val)
            is_null = builder.zext(_get_is_null(builder, iv), lc.Type.int(64))
            len_ = builder.zext(iv.len, lc.Type.int(64))
            asint64 = builder.shl(len_, lc.Constant.int(lc.Type.int(64), 32))
            asint64 = builder.or_(asint64, is_null)
            asstructi64i8p = builder.insert_value(lc.Constant.undef(lc.Type.struct([lc.Type.int(64), lc.Type.pointer(lc.Type.int(8))])),
                                                  asint64,
                                                  0)
            asstructi64i8p = builder.insert_value(asstructi64i8p, iv.ptr, 1)
            return asstructi64i8p
        else:
            return val

    def get_abi_return_type(self, ty):
        # FIXME only work on x86-64 + gcc
        if ty == BooleanVal:
            return lc.Type.pointer(lc.Type.int(16))
        elif ty == TinyIntVal:
            return lc.Type.pointer(lc.Type.int(16))
        elif ty == SmallIntVal:
            return lc.Type.pointer(lc.Type.int(32))
        elif ty == IntVal:
            return lc.Type.pointer(lc.Type.int(64))
        elif ty == BigIntVal:
            return lc.Type.pointer(lc.Type.struct([lc.Type.int(8), lc.Type.int(64)]))
        elif ty == FloatVal:
            return lc.Type.pointer(lc.Type.int(64))
        elif ty == DoubleVal:
            return lc.Type.pointer(lc.Type.struct([lc.Type.int(8), lc.Type.double()]))
        elif ty == StringVal:
            return lc.Type.pointer(lc.Type.struct([lc.Type.int(64), lc.Type.pointer(lc.Type.int(8))]))
        else:
            return self.context.get_return_type(ty)

    def get_abi_argument_type(self, ty):
        return self.context.get_argument_type(ty)
