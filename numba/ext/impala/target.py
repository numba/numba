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

class AnyValStruct(cgutils.Structure):
    _fields = [('is_null', types.boolean)]


@implement('is', AnyVal, types.none)
def isnone_anyval(context, builder, sig, args):
    [x, y] = args
    val = AnyValStruct(context, builder, value=x)
    return val.is_null


def _get_is_null_pointer(builder, val):
    ptr = cgutils.inbound_gep(builder, val._getpointer(), 0, 0, 0)
    return ptr


def _get_is_null(builder, val):
    byte = builder.load(_get_is_null_pointer(builder, val))
    return builder.trunc(byte, lc.Type.int(1))


def _set_is_null(builder, val, is_null):
    byte = builder.zext(is_null, lc.Type.int(8))
    builder.store(byte, _get_is_null_pointer(builder, val))


def _get_val_pointer(builder, val):
    ptr = cgutils.inbound_gep(builder, val._getpointer(), 0, 1)
    return ptr

def _get_val(builder, val):
    raw_val = builder.load(_get_val_pointer(builder, val))
    return raw_val

def _set_val(builder, val, to):
    builder.store(to, _get_val_pointer(builder, val))


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


class StringValStruct(cgutils.Structure):
    _fields = [('parent',  AnyVal),
               ('len',     types.int32),
               ('ptr',     types.CPointer(types.uint8))]


@impl_attribute(StringVal, "is_null", types.boolean)
def stringval_is_null(context, builder, typ, value):
    """
    StringVal::is_null
    """
    iv = StringValStruct(context, builder, value=value)
    is_null = _get_is_null(builder, iv)
    return is_null

@impl_attribute(StringVal, "len", types.int32)
def stringval_len(context, builder, typ, value):
    """
    StringVal::len
    """
    iv = StringValStruct(context, builder, value=value)
    return iv.len

@impl_attribute(StringVal, "ptr", types.CPointer(types.uint8))
def stringval_ptr(context, builder, typ, value):
    """
    StringVal::ptr
    """
    iv = StringValStruct(context, builder, value=value)
    return iv.ptr

@impl_attribute(StringValType, "null", StringVal)
def stringval_null(context, builder, typ, value):
    """
    StringVal::null
    """
    iv = StringValStruct(context, builder)
    _set_is_null(builder, iv, cgutils.true_bit)
    return iv._getvalue()

@implement(types.len_type, StringVal)
def len_stringval(context, builder, sig, args):
    [s] = args
    val = StringValStruct(context, builder, value=s)
    return val.len

@implement("==", StringVal, StringVal)
def eq_stringval(context, builder, sig, args):
    import ipdb
    ipdb.set_trace()
    [s1, s2] = args
    sv1 = StringValStruct(context, builder, value=s1)
    sv2 = StringValStruct(context, builder, value=s2)
    pass
    # TODO

@implement("==", types.CPointer(types.uint8), types.CPointer(types.uint8))
def eq_pointeruint8(context, builder, sig, args):
    [p1, p2] = args
    return builder.icmp(lc.ICMP_EQ, p1, p2)

@implement("getitem", StringVal, types.int64)
def getitem_stringval(context, builder, sig, args):
    [s, i] = args
    # TODO: check that the requested element is within the allocated String
    val = StringValStruct(context, builder, value=s)
    dataptr = cgutils.inbound_gep(builder, val.ptr, i)
    # THIS IS INCORRECT.  We must actually allocate some memory by calling the StringVal constructor
    elt = StringValStruct(context, builder)
    _set_is_null(builder, elt, cgutils.false_bit)
    iv.val
    return builder.load(dataptr)

    # vt = self.get_value_type(ty)
    # tmp = cgutils.alloca_once(builder, vt)
    # dataptr = cgutils.inbound_gep(builder, ptr, 0, 0)
    # builder.store(dataptr, cgutils.inbound_gep(builder, tmp, 0, 0))
    # return builder.load(tmp)

    # def inbound_gep(builder, ptr, *inds):
    #     idx = []
    #     for i in inds:
    #         if isinstance(i, int):
    #             ind = Constant.int(Type.int(32), i)
    #         else:
    #             ind = i
    #         idx.append(ind)
    #     return builder.gep(ptr, idx, inbounds=True)

    # %val = getelementptr inbounds %"struct.impala_udf::IntVal"* %arg2, i64 0, i32 1
    # %0 = load i32* %val, align 4, !tbaa !4
    # %idxprom = sext i32 %0 to i64
    # %ptr = getelementptr inbounds %"struct.impala_udf::StringVal"* %arg1, i64 0, i32 2
    # %1 = load i8** %ptr, align 8, !tbaa !5
    # %arrayidx = getelementptr inbounds i8* %1, i64 %idxprom
    # %2 = load i8* %arrayidx, align 1, !tbaa !1
    # ret i8 %2


@implement('StringValToInt16', StringVal)
def stringval_to_int16(context, builder, sig, args):
    # TODO: insert test so that StringVal must have len=1
    [s] = args
    iv = StringValStruct(context, builder, value=s)
    dataptr = cgutils.inbound_gep(builder, iv.ptr, 0)
    return builder.load(dataptr)


@implement(StringValType, types.CPointer(types.uint8), types.int32)
def stringval_ctor1(context, builder, sig, args):
    """
    StringVal(uint8_t* ptr, int32 len)
    """
    [x, y] = args
    iv = StringValStruct(context, builder)
    _set_is_null(builder, iv, cgutils.false_bit)
    iv.ptr = x
    iv.len = y
    return iv._getvalue()

@implement(StringValType, types.string)
def stringval_ctor2(context, builder, sig, args):
    """
    StringVal(types.string)
    """
    import ipdb
    ipdb.set_trace()
    [x] = args
    iv = StringValStruct(context, builder)
    _set_is_null(builder, iv, cgutils.false_bit)
    fndesc = lowering.describe_external('strlen', types.uintp, [types.CPointer(types.char)])
    func = context.declare_extern_c_function(cgutils.get_module(builder), fndesc)
    strlen_x = context.call_extern_c_function(builder, func, fndesc.argtypes, [x])
    len_x = builder.trunc(strlen_x, lc.Type.int(32))
    iv.len = len_x
    iv.ptr = x
    return iv._getvalue()


# @implement(StringValType, types.CPointer(types.char))
# def stringval_ctor2(context, builder, sig, args):
#     """
#     StringVal(const char* ptr)
#     """
#     [x, y] = args
#     iv = StringValStruct(context, builder)
#     _set_is_null(builder, iv, cgutils.false_bit)
#     iv.ptr = x
#     iv.len = y
#     return iv._getvalue()

# @implement(StringValType, types.CPointer(FunctionContext), types.int32)
# def stringval_ctor3(context, builder, sig, args):
#     """
#     StringVal(FunctionContext*, int32)
#     """
#     [x, y] = args
#     iv = StringValStruct(context, builder)
#     _set_is_null(builder, iv, cgutils.false_bit)
#     iv.ptr = x.
#     iv.len = y
#     return iv._getvalue()


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
        self.insert_attr_defn([booleanval_is_null, booleanval_val, booleanval_null,
                               tinyintval_is_null, tinyintval_val, tinyintval_null,
                               smallintval_is_null, smallintval_val, smallintval_null,
                               intval_is_null, intval_val, intval_null,
                               bigintval_is_null, bigintval_val, bigintval_null,
                               floatval_is_null, floatval_val, floatval_null,
                               doubleval_is_null, doubleval_val, doubleval_null,
                               stringval_is_null, stringval_len, stringval_ptr, stringval_null])
        self.insert_func_defn([booleanval_ctor, tinyintval_ctor,
                               smallintval_ctor, intval_ctor, bigintval_ctor,
                               floatval_ctor, doubleval_ctor, stringval_ctor1, stringval_ctor2,
                               len_stringval, isnone_anyval, getitem_stringval, stringval_to_int16, eq_pointeruint8, eq_stringval])
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
            raw_val = _get_val(builder, BooleanValStruct(self, builder, val))
            return self.cast(builder, raw_val, types.boolean, toty)
        if fromty == TinyIntVal:
            raw_val = _get_val(builder, TinyIntValStruct(self, builder, val))
            return self.cast(builder, raw_val, types.int8, toty)
        if fromty == SmallIntVal:
            raw_val = _get_val(builder, SmallIntValStruct(self, builder, val))
            return self.cast(builder, raw_val, types.int16, toty)
        if fromty == IntVal:
            raw_val = _get_val(builder, IntValStruct(self, builder, val))
            return self.cast(builder, raw_val, types.int32, toty)
        if fromty == BigIntVal:
            raw_val = _get_val(builder, BigIntValStruct(self, builder, val))
            return self.cast(builder, raw_val, types.int64, toty)
        if fromty == FloatVal:
            raw_val = _get_val(builder, FloatValStruct(self, builder, val))
            return self.cast(builder, raw_val, types.float32, toty)
        if fromty == DoubleVal:
            raw_val = _get_val(builder, DoubleValStruct(self, builder, val))
            return self.cast(builder, raw_val, types.float64, toty)

        # no way fromty is a *Val starting here
        if toty == BooleanVal:
            val = super(ImpalaTargetContext, self).cast(builder, val, fromty, types.int8)
            iv = BooleanValStruct(self, builder)
            _set_is_null(builder, iv, cgutils.false_bit)
            _set_val(builder, iv, val)
            return iv._getvalue()
        if toty == TinyIntVal:
            val = super(ImpalaTargetContext, self).cast(builder, val, fromty, types.int8)
            iv = TinyIntValStruct(self, builder)
            _set_is_null(builder, iv, cgutils.false_bit)
            _set_val(builder, iv, val)
            return iv._getvalue()
        if toty == SmallIntVal:
            val = super(ImpalaTargetContext, self).cast(builder, val, fromty, types.int16)
            iv = SmallIntValStruct(self, builder)
            _set_is_null(builder, iv, cgutils.false_bit)
            _set_val(builder, iv, val)
            return iv._getvalue()
        if toty == IntVal:
            val = super(ImpalaTargetContext, self).cast(builder, val, fromty, types.int32)
            iv = IntValStruct(self, builder)
            _set_is_null(builder, iv, cgutils.false_bit)
            _set_val(builder, iv, val)
            return iv._getvalue()
        if toty == BigIntVal:
            val = super(ImpalaTargetContext, self).cast(builder, val, fromty, types.int64)
            iv = BigIntValStruct(self, builder)
            _set_is_null(builder, iv, cgutils.false_bit)
            _set_val(builder, iv, val)
            return iv._getvalue()
        if toty == FloatVal:
            val = super(ImpalaTargetContext, self).cast(builder, val, fromty, types.float32)
            iv = FloatValStruct(self, builder)
            _set_is_null(builder, iv, cgutils.false_bit)
            _set_val(builder, iv, val)
            return iv._getvalue()
        if toty == DoubleVal:
            val = super(ImpalaTargetContext, self).cast(builder, val, fromty, types.float64)
            iv = DoubleValStruct(self, builder)
            _set_is_null(builder, iv, cgutils.false_bit)
            _set_val(builder, iv, val)
            return iv._getvalue()
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
