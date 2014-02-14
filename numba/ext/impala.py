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
IntVal = ImpalaValue('IntVal')
IntValType = types.Dummy('IntValType')


class IntValCtor(ConcreteTemplate):
    key = IntValType
    cases = [signature(IntVal, types.int32)]


class ImpalaValueAttr(AttributeTemplate):
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


class ImpalaTypeAttr(AttributeTemplate):
    key = IntValType

    def resolve_null(self, typ):
        """
        IntVal::null
        """
        return IntVal


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
    base.insert_global(IntVal, IntValType)
    base.insert_function(IntValCtor(base))
    base.insert_attributes(ImpalaValueAttr(base))
    base.insert_attributes(ImpalaTypeAttr(base))
    return base


#---------------------------------------------------------------------------
# Target implementation

class AnyValStruct(cgutils.Structure):
    _fields = [('is_null', types.boolean)]


class IntValStruct(cgutils.Structure):
    _fields = [('parent',  AnyVal),
               ('val',     types.int32),]


def _get_is_null_pointer(builder, val):
    ptr = cgutils.inbound_gep(builder, val._getpointer(), 0, 0, 0)
    return ptr


def _get_is_null(builder, val):
    byte = builder.load(_get_is_null_pointer(builder, val))
    return builder.trunc(byte, lc.Type.int(1))


def _set_is_null(builder, val, is_null):
    byte = builder.zext(is_null, lc.Type.int(8))
    builder.store(byte, _get_is_null_pointer(builder, val))


@impl_attribute(IntVal, "is_null", types.boolean)
def intval_is_null(context, builder, typ, value):
    """
    IntVall::is_null
    """
    iv = IntValStruct(context, builder, value=value)
    is_null = _get_is_null(builder, iv)
    return is_null

@impl_attribute(IntVal, "val", types.int32)
def intval_val(context, builder, typ, value):
    """
    IntVall::val
    """
    iv = IntValStruct(context, builder, value=value)
    return iv.val


@impl_attribute(IntValType, "null", IntVal)
def intval_null(context, builder, typ, value):
    """
    IntVall::null
    """
    iv = IntValStruct(context, builder)
    _set_is_null(builder, iv, cgutils.true_bit)
    return iv._getvalue()


@implement(IntValType, types.int32)
def intval_ctor(context, builder, sig, args):
    """
    IntVall(int32)
    """
    [x] = args
    iv = IntValStruct(context, builder)
    _set_is_null(builder, iv, cgutils.false_bit)
    iv.val = x
    return iv._getvalue()


TYPE_LAYOUT = {
    AnyVal: AnyValStruct,
    IntVal: IntValStruct,
}


class ImpalaTargetContext(BaseContext):
    def init(self):
        self.tm = le.TargetMachine.new()
        self.insert_attr_defn([intval_is_null, intval_val,
                               intval_null])
        self.insert_func_defn([intval_ctor])
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
        if ty == IntVal:
            # Pack structure into int64
            # Endian specific
            iv = IntValStruct(self.context, builder, value=val)
            lower = builder.zext(_get_is_null(builder, iv), lc.Type.int(64))
            upper = builder.zext(iv.val, lc.Type.int(64))

            asint64 = builder.shl(upper, lc.Constant.int(lc.Type.int(64), 32))
            asint64 = builder.or_(asint64, lower)

            return asint64
        else:
            return val

    def get_abi_return_type(self, ty):
        # FIXME only work on x86-64 + gcc
        if ty == IntVal:
            return lc.Type.int(64)
        else:
            return self.context.get_return_type(ty)
        return self.context.get_return_type(ty)

    def get_abi_argument_type(self, ty):
        return self.context.get_argument_type(ty)


#---------------------------------------------------------------------------
# Target description

impala_typing = impala_typing_context()
impala_targets = ImpalaTargetContext(impala_typing)

