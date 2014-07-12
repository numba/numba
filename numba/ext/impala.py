from __future__ import print_function, division, absolute_import
import llvm.core as lc
import llvm.passes as lp
import llvm.ee as le
from numba.compiler import compile_extra, Flags
from numba import typing, sigutils, types, cgutils, config
from numba.targets.base import BaseContext
from numba.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                    signature)
from numba.targets.imputils import implement, impl_attribute


def udf(signature):
    def wrapper(pyfunc):
        udfobj = UDF(pyfunc, signature)
        return udfobj
    return wrapper


#---------------------------------------------------------------------------
# Typing information

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


class UDF(object):
    def __init__(self, pyfunc, signature):
        self.py_func = pyfunc
        self.signature = signature
        self.name = pyfunc.__name__

        args, return_type = sigutils.normalize_signature(signature)
        flags = Flags()
        flags.set('no_compile')
        self._cres = compile_extra(typingctx=impala_typing,
                                   targetctx=impala_targets, func=pyfunc,
                                   args=args, return_type=return_type,
                                   flags=flags, locals={})
        llvm_func = impala_targets.finalize(self._cres.llvm_func, return_type,
                                            args)
        self.llvm_func = llvm_func
        self.llvm_module = llvm_func.module


def impala_typing_context():
    base = typing.Context()
    
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
    
    return base


#---------------------------------------------------------------------------
# Target implementation

class AnyValStruct(cgutils.Structure):
    _fields = [('is_null', types.boolean)]


def _get_is_null_pointer(builder, val):
    ptr = cgutils.inbound_gep(builder, val._getpointer(), 0, 0, 0)
    return ptr


def _get_is_null(builder, val):
    byte = builder.load(_get_is_null_pointer(builder, val))
    return builder.trunc(byte, lc.Type.int(1))


def _set_is_null(builder, val, is_null):
    byte = builder.zext(is_null, lc.Type.int(8))
    builder.store(byte, _get_is_null_pointer(builder, val))


class BooleanValStruct(cgutils.Structure):
    _fields = [('parent',  AnyVal),
               ('val',     types.int8),]


@impl_attribute(BooleanVal, "is_null", types.boolean)
def booleanval_is_null(context, builder, typ, value):
    """
    BooleanVal::is_null
    """
    iv = BooleanValStruct(context, builder, value=value)
    is_null = _get_is_null(builder, iv)
    return is_null

@impl_attribute(BooleanVal, "val", types.int8)
def booleanval_val(context, builder, typ, value):
    """
    BooleanVal::val
    """
    iv = BooleanValStruct(context, builder, value=value)
    return iv.val


@impl_attribute(BooleanValType, "null", BooleanVal)
def booleanval_null(context, builder, typ, value):
    """
    BooleanVal::null
    """
    iv = BooleanValStruct(context, builder)
    _set_is_null(builder, iv, cgutils.true_bit)
    return iv._getvalue()


@implement(BooleanValType, types.int8)
def booleanval_ctor(context, builder, sig, args):
    """
    BooleanVal(int8)
    """
    [x] = args
    iv = BooleanValStruct(context, builder)
    _set_is_null(builder, iv, cgutils.false_bit)
    iv.val = x
    return iv._getvalue()


class TinyIntValStruct(cgutils.Structure):
    _fields = [('parent',  AnyVal),
               ('val',     types.int8),]


@impl_attribute(TinyIntVal, "is_null", types.boolean)
def tinyintval_is_null(context, builder, typ, value):
    """
    TinyIntVal::is_null
    """
    iv = TinyIntValStruct(context, builder, value=value)
    is_null = _get_is_null(builder, iv)
    return is_null

@impl_attribute(TinyIntVal, "val", types.int8)
def tinyintval_val(context, builder, typ, value):
    """
    TinyIntVal::val
    """
    iv = TinyIntValStruct(context, builder, value=value)
    return iv.val


@impl_attribute(TinyIntValType, "null", TinyIntVal)
def tinyintval_null(context, builder, typ, value):
    """
    TinyIntVal::null
    """
    iv = TinyIntValStruct(context, builder)
    _set_is_null(builder, iv, cgutils.true_bit)
    return iv._getvalue()


@implement(TinyIntValType, types.int8)
def tinyintval_ctor(context, builder, sig, args):
    """
    TinyIntVal(int8)
    """
    [x] = args
    iv = TinyIntValStruct(context, builder)
    _set_is_null(builder, iv, cgutils.false_bit)
    iv.val = x
    return iv._getvalue()


class SmallIntValStruct(cgutils.Structure):
    _fields = [('parent',  AnyVal),
               ('val',     types.int16),]


@impl_attribute(SmallIntVal, "is_null", types.boolean)
def smallintval_is_null(context, builder, typ, value):
    """
    SmallIntVal::is_null
    """
    iv = SmallIntValStruct(context, builder, value=value)
    is_null = _get_is_null(builder, iv)
    return is_null

@impl_attribute(SmallIntVal, "val", types.int16)
def smallintval_val(context, builder, typ, value):
    """
    SmallIntVal::val
    """
    iv = SmallIntValStruct(context, builder, value=value)
    return iv.val


@impl_attribute(SmallIntValType, "null", SmallIntVal)
def smallintval_null(context, builder, typ, value):
    """
    SmallIntVal::null
    """
    iv = SmallIntValStruct(context, builder)
    _set_is_null(builder, iv, cgutils.true_bit)
    return iv._getvalue()


@implement(SmallIntValType, types.int16)
def smallintval_ctor(context, builder, sig, args):
    """
    SmallIntVal(int16)
    """
    [x] = args
    iv = SmallIntValStruct(context, builder)
    _set_is_null(builder, iv, cgutils.false_bit)
    iv.val = x
    return iv._getvalue()


class IntValStruct(cgutils.Structure):
    _fields = [('parent',  AnyVal),
               ('val',     types.int32),]


@impl_attribute(IntVal, "is_null", types.boolean)
def intval_is_null(context, builder, typ, value):
    """
    IntVal::is_null
    """
    iv = IntValStruct(context, builder, value=value)
    is_null = _get_is_null(builder, iv)
    return is_null

@impl_attribute(IntVal, "val", types.int32)
def intval_val(context, builder, typ, value):
    """
    IntVal::val
    """
    iv = IntValStruct(context, builder, value=value)
    return iv.val


@impl_attribute(IntValType, "null", IntVal)
def intval_null(context, builder, typ, value):
    """
    IntVal::null
    """
    iv = IntValStruct(context, builder)
    _set_is_null(builder, iv, cgutils.true_bit)
    return iv._getvalue()


@implement(IntValType, types.int32)
def intval_ctor(context, builder, sig, args):
    """
    IntVal(int32)
    """
    [x] = args
    iv = IntValStruct(context, builder)
    _set_is_null(builder, iv, cgutils.false_bit)
    iv.val = x
    return iv._getvalue()


class BigIntValStruct(cgutils.Structure):
    _fields = [('parent',  AnyVal),
               ('val',     types.int64),]


@impl_attribute(BigIntVal, "is_null", types.boolean)
def bigintval_is_null(context, builder, typ, value):
    """
    BigIntVal::is_null
    """
    iv = BigIntValStruct(context, builder, value=value)
    is_null = _get_is_null(builder, iv)
    return is_null

@impl_attribute(BigIntVal, "val", types.int64)
def bigintval_val(context, builder, typ, value):
    """
    BigIntVal::val
    """
    iv = BigIntValStruct(context, builder, value=value)
    return iv.val


@impl_attribute(BigIntValType, "null", BigIntVal)
def bigintval_null(context, builder, typ, value):
    """
    BigIntVal::null
    """
    iv = BigIntValStruct(context, builder)
    _set_is_null(builder, iv, cgutils.true_bit)
    return iv._getvalue()


@implement(BigIntValType, types.int64)
def bigintval_ctor(context, builder, sig, args):
    """
    BigIntVal(int64)
    """
    [x] = args
    iv = BigIntValStruct(context, builder)
    _set_is_null(builder, iv, cgutils.false_bit)
    iv.val = x
    return iv._getvalue()


class FloatValStruct(cgutils.Structure):
    _fields = [('parent',  AnyVal),
               ('val',     types.float32),]


@impl_attribute(FloatVal, "is_null", types.boolean)
def floatval_is_null(context, builder, typ, value):
    """
    FloatVal::is_null
    """
    iv = FloatValStruct(context, builder, value=value)
    is_null = _get_is_null(builder, iv)
    return is_null

@impl_attribute(FloatVal, "val", types.float32)
def floatval_val(context, builder, typ, value):
    """
    FloatVal::val
    """
    iv = FloatValStruct(context, builder, value=value)
    return iv.val


@impl_attribute(FloatValType, "null", FloatVal)
def floatval_null(context, builder, typ, value):
    """
    FloatVal::null
    """
    iv = FloatValStruct(context, builder)
    _set_is_null(builder, iv, cgutils.true_bit)
    return iv._getvalue()


@implement(FloatValType, types.float32)
def floatval_ctor(context, builder, sig, args):
    """
    FloatVal(float32)
    """
    [x] = args
    iv = FloatValStruct(context, builder)
    _set_is_null(builder, iv, cgutils.false_bit)
    iv.val = x
    return iv._getvalue()


class DoubleValStruct(cgutils.Structure):
    _fields = [('parent',  AnyVal),
               ('val',     types.float64),]


@impl_attribute(DoubleVal, "is_null", types.boolean)
def doubleval_is_null(context, builder, typ, value):
    """
    DoubleVal::is_null
    """
    iv = DoubleValStruct(context, builder, value=value)
    is_null = _get_is_null(builder, iv)
    return is_null

@impl_attribute(DoubleVal, "val", types.float64)
def doubleval_val(context, builder, typ, value):
    """
    DoubleVal::val
    """
    iv = DoubleValStruct(context, builder, value=value)
    return iv.val


@impl_attribute(DoubleValType, "null", DoubleVal)
def doubleval_null(context, builder, typ, value):
    """
    DoubleVal::null
    """
    iv = DoubleValStruct(context, builder)
    _set_is_null(builder, iv, cgutils.true_bit)
    return iv._getvalue()


@implement(DoubleValType, types.float64)
def doubleval_ctor(context, builder, sig, args):
    """
    DoubleVal(float64)
    """
    [x] = args
    iv = DoubleValStruct(context, builder)
    _set_is_null(builder, iv, cgutils.false_bit)
    iv.val = x
    return iv._getvalue()

TYPE_LAYOUT = {
    AnyVal: AnyValStruct,
    BooleanVal: BooleanValStruct,
    TinyIntVal: TinyIntValStruct,
    SmallIntVal: SmallIntValStruct,
    IntVal: IntValStruct,
    BigIntVal: BigIntValStruct,
    FloatVal: FloatValStruct,
    DoubleVal: DoubleValStruct,
}


class ImpalaTargetContext(BaseContext):
    def init(self):
        self.tm = le.TargetMachine.new()
        self.insert_attr_defn([booleanval_is_null, booleanval_val, booleanval_null,
                               tinyintval_is_null, tinyintval_val, tinyintval_null,
                               smallintval_is_null, smallintval_val, smallintval_null,
                               intval_is_null, intval_val, intval_null,
                               bigintval_is_null, bigintval_val, bigintval_null,
                               floatval_is_null, floatval_val, floatval_null,
                               doubleval_is_null, doubleval_val, doubleval_null])
        self.insert_func_defn([booleanval_ctor, tinyintval_ctor,
                               smallintval_ctor, intval_ctor, bigintval_ctor,
                               floatval_ctor, doubleval_ctor])
        self.optimizer = self.build_pass_manager()

        # once per context
        self._fnctximpltype = lc.Type.opaque("FunctionContextImpl")
        fnctxbody = [lc.Type.pointer(self._fnctximpltype)]
        self._fnctxtype = lc.Type.struct(fnctxbody,
                                        name="class.impala_udf::FunctionContext")

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
        abi_restype = self.get_abi_return_type(self.restype)
        abi_argtypes = [self.get_abi_argument_type(a)
                        for a in self.argtypes]
        fnty = lc.Type.function(abi_restype, abi_argtypes)
        wrapper = self.func.module.add_function(fnty, name=wrappername)

        builder = lc.Builder.new(wrapper.append_basic_block(''))
        status, res = self.context.call_function(builder, self.func,
                                                 self.restype,
                                                 self.argtypes,
                                                 wrapper.args)
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
        else:
            return val

    def get_abi_return_type(self, ty):
        # FIXME only work on x86-64 + gcc
        if ty == BooleanVal:
            return lc.Type.int(16)
        elif ty == TinyIntVal:
            return lc.Type.int(16)
        elif ty == SmallIntVal:
            return lc.Type.int(32)
        elif ty == IntVal:
            return lc.Type.int(64)
        elif ty == BigIntVal:
            return lc.Type.struct([lc.Type.int(8), lc.Type.int(64)])
        elif ty == FloatVal:
            return lc.Type.int(64)
        elif ty == DoubleVal:
            return lc.Type.struct([lc.Type.int(8), lc.Type.double()])
        else:
            return self.context.get_return_type(ty)
        return self.context.get_return_type(ty)

    def get_abi_argument_type(self, ty):
        return self.context.get_argument_type(ty)


#---------------------------------------------------------------------------
# Target description

impala_typing = impala_typing_context()
impala_targets = ImpalaTargetContext(impala_typing)
