from __future__ import print_function
from collections import namedtuple, defaultdict
import copy
from types import MethodType

import numpy

from llvmlite import ir as llvmir
import llvmlite.llvmpy.core as lc
from llvmlite.llvmpy.core import Type, Constant

import numba
from numba import types, utils, cgutils, typing, numpy_support, errcode
from numba.pythonapi import PythonAPI
from numba.targets.imputils import (user_function, python_attr_impl,
                                    builtin_registry, impl_attribute,
                                    struct_registry, type_registry)
from . import arrayobj, builtins, iterators, rangeobj, optional
try:
    from . import npdatetime
except NotImplementedError:
    pass


GENERIC_POINTER = Type.pointer(Type.int(8))
PYOBJECT = GENERIC_POINTER

LTYPEMAP = {
    types.pyobject: PYOBJECT,

    types.boolean: Type.int(8),

    types.uint8: Type.int(8),
    types.uint16: Type.int(16),
    types.uint32: Type.int(32),
    types.uint64: Type.int(64),

    types.int8: Type.int(8),
    types.int16: Type.int(16),
    types.int32: Type.int(32),
    types.int64: Type.int(64),

    types.float32: Type.float(),
    types.float64: Type.double(),
}

STRUCT_TYPES = {
    types.complex64: builtins.Complex64,
    types.complex128: builtins.Complex128,
    types.slice3_type: builtins.Slice,
}

Status = namedtuple("Status", ("code", "ok", "err", "exc", "none"))

RETCODE_OK = Constant.int_signextend(Type.int(), 0)
RETCODE_NONE = Constant.int_signextend(Type.int(), -2)
RETCODE_EXC = Constant.int_signextend(Type.int(), -1)


class Overloads(object):
    def __init__(self):
        self.versions = []

    def find(self, sig):
        for i, ver in enumerate(self.versions):
            if ver.signature == sig:
                return ver

            # As generic type
            if self._match_arglist(ver.signature.args, sig.args):
                return ver

        raise NotImplementedError(self, sig)

    def _match_arglist(self, formal_args, actual_args):
        if formal_args and isinstance(formal_args[-1], types.VarArg):
            formal_args = (
                formal_args[:-1] +
                (formal_args[-1].dtype,) * (len(actual_args) - len(formal_args) + 1))

        if len(formal_args) != len(actual_args):
            return False

        for formal, actual in zip(formal_args, actual_args):
            if not self._match(formal, actual):
                return False

        return True

    def _match(self, formal, actual):
        if formal == actual:
            # formal argument matches actual arguments
            return True
        elif types.Any == formal:
            # formal argument is any
            return True
        elif (isinstance(formal, types.Kind) and
                  isinstance(actual, formal.of)):
            # formal argument is a kind and the actual argument
            # is of that kind
            return True

    def append(self, impl):
        self.versions.append(impl)


class BaseContext(object):
    """

    Notes on Structure
    ------------------

    Most objects are lowered as plain-old-data structure in the generated
    llvm.  They are passed around by reference (a pointer to the structure).
    Only POD structure can life across function boundaries by copying the
    data.
    """
    # True if the target requires strict alignment
    # Causes exception to be raised if the record members are not aligned.
    strict_alignment = False

    # Use default mangler (no specific requirement)
    mangler = None

    # Force powi implementation as math.pow call
    implement_powi_as_math_call = False
    implement_pow_as_math_call = False

    def __init__(self, typing_context):
        self.address_size = utils.MACHINE_BITS
        self.typing_context = typing_context

        self.defns = defaultdict(Overloads)
        self.attrs = defaultdict(Overloads)
        self.users = utils.UniqueDict()

        self.insert_func_defn(builtin_registry.functions)
        self.insert_attr_defn(builtin_registry.attributes)

        self.cached_internal_func = {}

        # Initialize
        self.init()

    def init(self):
        """
        For subclasses to add initializer
        """
        pass

    @property
    def target_data(self):
        raise NotImplementedError

    def insert_func_defn(self, defns):
        for defn in defns:
            self.defns[defn.key].append(defn)

    def insert_attr_defn(self, defns):
        for imp in defns:
            self.attrs[imp.attr].append(imp)

    def insert_user_function(self, func, fndesc, libs=()):
        imp = user_function(func, fndesc, libs)
        self.defns[func].append(imp)

        baseclses = (typing.templates.ConcreteTemplate,)
        glbls = dict(key=func, cases=[imp.signature])
        name = "CallTemplate(%s)" % fndesc.mangled_name
        self.users[func] = type(name, baseclses, glbls)

    def add_user_function(self, func, fndesc, libs=()):
        if func not in self.users:
            msg = "{func} is not a registered user function"
            raise KeyError(msg.format(func=func))
        imp = user_function(func, fndesc, libs)
        self.defns[func].append(imp)

    def insert_class(self, cls, attrs):
        clsty = types.Object(cls)
        for name, vtype in utils.iteritems(attrs):
            imp = python_attr_impl(clsty, name, vtype)
            self.attrs[imp.attr].append(imp)

    def remove_user_function(self, func):
        """
        Remove user function *func*.
        KeyError is raised if the function isn't known to us.
        """
        del self.users[func]
        del self.defns[func]

    def get_user_function(self, func):
        return self.users[func]

    def get_function_type(self, fndesc):
        """
        Get the implemented Function type for the high-level *fndesc*.
        Some parameters can be added or shuffled around.
        This is kept in sync with call_function() and get_arguments().

        Calling Convention
        ------------------
        Returns: -2 for return none in native function;
                 -1 for failure with python exception set;
                  0 for success;
                 >0 for user error code.
        Return value is passed by reference as the first argument.

        Actual arguments starts at the 2rd argument position.
        Caller is responsible to allocate space for return value.
        """
        argtypes = [self.get_argument_type(aty)
                    for aty in fndesc.argtypes]
        resptr = self.get_return_type(fndesc.restype)
        fnty = Type.function(Type.int(), [resptr] + argtypes)
        return fnty

    def get_external_function_type(self, fndesc):
        argtypes = [self.get_argument_type(aty)
                    for aty in fndesc.argtypes]
        # don't wrap in pointer
        restype = self.get_argument_type(fndesc.restype)
        fnty = Type.function(restype, argtypes)
        return fnty

    def declare_function(self, module, fndesc):
        fnty = self.get_function_type(fndesc)
        fn = module.get_or_insert_function(fnty, name=fndesc.mangled_name)
        assert fn.is_declaration
        for ak, av in zip(fndesc.args, self.get_arguments(fn)):
            av.name = "arg.%s" % ak
        fn.args[0].name = ".ret"
        if fndesc.inline:
            fn.attributes.add('alwaysinline')
        return fn

    def declare_external_function(self, module, fndesc):
        fnty = self.get_external_function_type(fndesc)
        fn = module.get_or_insert_function(fnty, name=fndesc.mangled_name)
        assert fn.is_declaration
        for ak, av in zip(fndesc.args, fn.args):
            av.name = "arg.%s" % ak
        return fn

    def insert_const_string(self, mod, string):
        stringtype = GENERIC_POINTER
        text = Constant.stringz(string)
        name = ".const.%s" % string
        for gv in mod.global_variables:
            if gv.name == name and gv.type.pointee == text.type:
                break
        else:
            gv = cgutils.global_constant(mod, name, text)
            gv.linkage = lc.LINKAGE_INTERNAL
        return Constant.bitcast(gv, stringtype)

    def get_arguments(self, func):
        """
        Get the Python-level arguments of LLVM *func*.
        See get_function_type() for the calling convention.
        """
        return func.args[1:]

    def get_argument_type(self, ty):
        if ty == types.boolean:
            return self.get_data_type(ty)
        elif self.is_struct_type(ty):
            return Type.pointer(self.get_value_type(ty))
        else:
            return self.get_value_type(ty)

    def get_return_type(self, ty):
        if isinstance(ty, types.Optional):
            return self.get_return_type(ty.type)
        elif self.is_struct_type(ty):
            return self.get_argument_type(ty)
        else:
            argty = self.get_argument_type(ty)
            return Type.pointer(argty)

    def get_data_type(self, ty):
        """
        Get a LLVM data representation of the Numba type *ty* that is safe
        for storage.  Record data are stored as byte array.

        The return value is a llvmlite.ir.Type object, or None if the type
        is an opaque pointer (???).
        """
        try:
            fac = type_registry.match(ty)
        except KeyError:
            pass
        else:
            return fac(self, ty)

        if (isinstance(ty, types.Dummy) or
                isinstance(ty, types.Module) or
                isinstance(ty, types.Function) or
                isinstance(ty, types.Dispatcher) or
                isinstance(ty, types.Object) or
                isinstance(ty, types.Macro)):
            return PYOBJECT

        elif isinstance(ty, types.CPointer):
            dty = self.get_data_type(ty.dtype)
            return Type.pointer(dty)

        elif isinstance(ty, types.Optional):
            return self.get_struct_type(self.make_optional(ty))

        elif isinstance(ty, types.Array):
            return self.get_struct_type(self.make_array(ty))

        elif isinstance(ty, types.UniTuple):
            dty = self.get_value_type(ty.dtype)
            return Type.array(dty, ty.count)

        elif isinstance(ty, types.Tuple):
            dtys = [self.get_value_type(t) for t in ty]
            return Type.struct(dtys)

        elif isinstance(ty, types.Record):
            # Record are represented as byte array
            return Type.struct([Type.array(Type.int(8), ty.size)])

        elif isinstance(ty, types.UnicodeCharSeq):
            charty = Type.int(numpy_support.sizeof_unicode_char * 8)
            return Type.struct([Type.array(charty, ty.count)])

        elif isinstance(ty, types.CharSeq):
            charty = Type.int(8)
            return Type.struct([Type.array(charty, ty.count)])

        elif ty in STRUCT_TYPES:
            return self.get_struct_type(STRUCT_TYPES[ty])

        else:
            try:
                impl = struct_registry.match(ty)
            except KeyError:
                pass
            else:
                return self.get_struct_type(impl(ty))

        if isinstance(ty, types.Pair):
            pairty = self.make_pair(ty.first_type, ty.second_type)
            return self.get_struct_type(pairty)

        else:
            return LTYPEMAP[ty]

    def get_value_type(self, ty):
        if ty == types.boolean:
            return Type.int(1)
        dataty = self.get_data_type(ty)

        if isinstance(ty, types.Record):
            # Record data are passed by refrence
            memory = dataty.elements[0]
            return Type.struct([Type.pointer(memory)])

        return dataty

    def pack_value(self, builder, ty, value, ptr):
        """Pack data for array storage
        """
        if isinstance(ty, types.Record):
            pdata = cgutils.get_record_data(builder, value)
            databuf = builder.load(pdata)
            casted = builder.bitcast(ptr, Type.pointer(databuf.type))
            builder.store(databuf, casted)
            return

        if ty == types.boolean:
            value = cgutils.as_bool_byte(builder, value)
        assert value.type == ptr.type.pointee
        builder.store(value, ptr)

    def unpack_value(self, builder, ty, ptr):
        """Unpack data from array storage
        """

        if isinstance(ty, types.Record):
            vt = self.get_value_type(ty)
            tmp = cgutils.alloca_once(builder, vt)
            dataptr = cgutils.inbound_gep(builder, ptr, 0, 0)
            builder.store(dataptr, cgutils.inbound_gep(builder, tmp, 0, 0))
            return builder.load(tmp)

        assert cgutils.is_pointer(ptr.type)
        value = builder.load(ptr)
        if ty == types.boolean:
            return builder.trunc(value, Type.int(1))
        else:
            return value

    def is_struct_type(self, ty):
        return cgutils.is_struct(self.get_data_type(ty))

    def get_constant_generic(self, builder, ty, val):
        """
        Return a LLVM constant representing value *val* of Numba type *ty*.
        """
        if self.is_struct_type(ty):
            return self.get_constant_struct(builder, ty, val)
        elif ty == types.string:
            return self.get_constant_string(builder, ty, val)
        else:
            return self.get_constant(ty, val)

    def get_constant_struct(self, builder, ty, val):
        assert self.is_struct_type(ty)
        module = cgutils.get_module(builder)

        if ty in types.complex_domain:
            if ty == types.complex64:
                innertype = types.float32
            elif ty == types.complex128:
                innertype = types.float64
            else:
                raise Exception("unreachable")

            real = self.get_constant(innertype, val.real)
            imag = self.get_constant(innertype, val.imag)
            const = Constant.struct([real, imag])
            return const

        elif isinstance(ty, types.Tuple):
            consts = [self.get_constant_generic(builder, ty.types[i], v)
                      for i, v in enumerate(val)]
            return Constant.struct(consts)

        elif isinstance(ty, types.Record):
            consts = [self.get_constant(types.int8, b)
                      for b in bytearray(val.tostring())]
            return Constant.array(consts[0].type, consts)

        else:
            raise NotImplementedError("%s as constant unsupported" % ty)

    def get_constant(self, ty, val):
        assert not self.is_struct_type(ty)

        lty = self.get_value_type(ty)

        if ty == types.none:
            assert val is None
            return self.get_dummy_value()

        elif ty == types.boolean:
            return Constant.int(Type.int(1), int(val))

        elif ty in types.signed_domain:
            return Constant.int_signextend(lty, val)

        elif ty in types.unsigned_domain:
            return Constant.int(lty, val)

        elif ty in types.real_domain:
            return Constant.real(lty, val)

        elif isinstance(ty, types.UniTuple):
            consts = [self.get_constant(ty.dtype, v) for v in val]
            return Constant.array(consts[0].type, consts)

        raise NotImplementedError(ty)

    def get_constant_undef(self, ty):
        lty = self.get_value_type(ty)
        return Constant.undef(lty)

    def get_constant_null(self, ty):
        lty = self.get_value_type(ty)
        return Constant.null(lty)

    def get_setattr(self, attr, sig):
        typ = sig.args[0]
        if isinstance(typ, types.Record):
            self.sentry_record_alignment(typ, attr)

            offset = typ.offset(attr)
            elemty = typ.typeof(attr)

            def imp(context, builder, sig, args):
                valty = sig.args[1]
                [target, val] = args
                dptr = cgutils.get_record_member(builder, target, offset,
                                                 self.get_data_type(elemty))
                val = context.cast(builder, val, valty, elemty)
                self.pack_value(builder, elemty, val, dptr)

            return _wrap_impl(imp, self, sig)

    def get_function(self, fn, sig):
        if isinstance(fn, types.Function):
            key = fn.template.key

            if isinstance(key, MethodType):
                overloads = self.defns[key.im_func]

            elif sig.recvr:
                sig = typing.signature(sig.return_type,
                                       *((sig.recvr,) + sig.args))
                overloads = self.defns[key]
            else:
                overloads = self.defns[key]

        elif isinstance(fn, types.Dispatcher):
            key = fn.overloaded.get_overload(sig.args)
            overloads = self.defns[key]
        else:
            key = fn
            overloads = self.defns[key]
        try:
            return _wrap_impl(overloads.find(sig), self, sig)
        except NotImplementedError:
            raise Exception("No definition for lowering %s%s" % (key, sig))

    def get_bound_function(self, builder, obj, ty):
        return obj

    def get_attribute(self, val, typ, attr):
        if isinstance(typ, types.Record):
            # Implement get attribute for records
            self.sentry_record_alignment(typ, attr)
            offset = typ.offset(attr)
            elemty = typ.typeof(attr)

            @impl_attribute(typ, attr, elemty)
            def imp(context, builder, typ, val):
                dptr = cgutils.get_record_member(builder, val, offset,
                                                 self.get_data_type(elemty))
                return self.unpack_value(builder, elemty, dptr)
            return imp

        if isinstance(typ, types.Module):
            # Implement getattr for module-level globals.
            # We are treating them as constants.
            # XXX We shouldn't have to retype this
            attrty = self.typing_context.resolve_module_constants(typ, attr)
            if attrty is not None:
                try:
                    pyval = getattr(typ.pymod, attr)
                    llval = self.get_constant(attrty, pyval)
                except NotImplementedError:
                    # Module attribute is not a simple constant
                    # (e.g. it's a function), it will be handled later on.
                    pass
                else:
                    @impl_attribute(typ, attr, attrty)
                    def imp(context, builder, typ, val):
                        return llval
                    return imp
            # No implementation
            return None

        # Lookup specific attribute implementation for this type
        overloads = self.attrs[attr]
        try:
            return overloads.find(typing.signature(types.Any, typ))
        except NotImplementedError:
            pass
        # Lookup generic getattr implementation for this type
        overloads = self.attrs[None]
        try:
            return overloads.find(typing.signature(types.Any, typ))
        except NotImplementedError:
            raise Exception("No definition for lowering %s.%s" % (typ, attr))

    def get_argument_value(self, builder, ty, val):
        """
        Argument representation to local value representation
        """
        if ty == types.boolean:
            return builder.trunc(val, self.get_value_type(ty))
        elif cgutils.is_struct_ptr(val.type):
            return builder.load(val)
        return val

    def get_returned_value(self, builder, ty, val):
        """
        Return value representation to local value representation
        """
        # Same as get_argument_value
        return self.get_argument_value(builder, ty, val)

    def get_return_value(self, builder, ty, val):
        """
        Local value representation to return type representation
        """

        if ty is types.boolean:
            r = self.get_return_type(ty).pointee
            return builder.zext(val, r)
        else:
            return val

    def get_struct_member_value(self, builder, ty, val):
        """
        Local value representation to struct member representation
        """
        # Currently same as get_return_value.
        # TODO: make a "TypeLayout" class that describe the layout at
        #       different situations for a value of a type.
        return self.get_return_value(builder, ty, val)

    def get_value_as_argument(self, builder, ty, val):
        """Prepare local value representation as argument type representation
        """
        argty = self.get_argument_type(ty)
        if argty == val.type:
            return val

        elif self.is_struct_type(ty):
            # Arguments are passed by pointer
            assert argty.pointee == val.type
            tmp = cgutils.alloca_once(builder, val.type)
            builder.store(val, tmp)
            return tmp

        elif ty == types.boolean:
            return builder.zext(val, argty)

        raise NotImplementedError("value %s -> arg %s" % (val.type, argty))

    def return_optional_value(self, builder, retty, valty, value):
        if valty == types.none:
            self.return_native_none(builder)

        elif retty == valty:
            optcls = self.make_optional(retty)
            optval = optcls(self, builder, value=value)

            validbit = builder.trunc(optval.valid, lc.Type.int(1))
            with cgutils.ifthen(builder, validbit):
                self.return_value(builder, optval.data)

            self.return_native_none(builder)

        elif not isinstance(valty, types.Optional):
            if valty != retty.type:
                value = self.cast(builder, value, fromty=valty,
                                  toty=retty.type)
            self.return_value(builder, value)

        else:
            raise NotImplementedError("returning {0} for {1}".format(valty,
                                                                     retty))

    def return_value(self, builder, retval):
        fn = cgutils.get_function(builder)
        retptr = fn.args[0]
        assert retval.type == retptr.type.pointee, \
            (str(retval.type), str(retptr.type.pointee))
        builder.store(retval, retptr)
        builder.ret(RETCODE_OK)

    def return_native_none(self, builder):
        builder.ret(RETCODE_NONE)

    def return_errcode(self, builder, code):
        assert code > 0
        builder.ret(Constant.int(Type.int(), code))

    def return_errcode_propagate(self, builder, code):
        builder.ret(code)

    def return_exc(self, builder):
        builder.ret(RETCODE_EXC)

    def return_user_exc(self, builder, code):
        assert code > 0
        builder.ret(Constant.int(Type.int(), code))

    def pair_first(self, builder, val, ty):
        """
        Extract the first element of a heterogenous pair.
        """
        paircls = self.make_pair(ty.first_type, ty.second_type)
        pair = paircls(self, builder, value=val)
        return self.get_argument_value(builder, ty.first_type, pair.first)

    def pair_second(self, builder, val, ty):
        """
        Extract the second element of a heterogenous pair.
        """
        paircls = self.make_pair(ty.first_type, ty.second_type)
        pair = paircls(self, builder, value=val)
        return self.get_argument_value(builder, ty.second_type, pair.second)

    def cast(self, builder, val, fromty, toty):
        if fromty == toty or toty == types.Any or isinstance(toty, types.Kind):
            return val

        elif ((fromty in types.unsigned_domain and
                       toty in types.signed_domain) or
                  (fromty in types.integer_domain and
                           toty in types.unsigned_domain)):
            lfrom = self.get_value_type(fromty)
            lto = self.get_value_type(toty)
            if lfrom.width <= lto.width:
                return builder.zext(val, lto)
            elif lfrom.width > lto.width:
                return builder.trunc(val, lto)

        elif fromty in types.signed_domain and toty in types.signed_domain:
            lfrom = self.get_value_type(fromty)
            lto = self.get_value_type(toty)
            if lfrom.width <= lto.width:
                return builder.sext(val, lto)
            elif lfrom.width > lto.width:
                return builder.trunc(val, lto)

        elif fromty in types.real_domain and toty in types.real_domain:
            lty = self.get_value_type(toty)
            if fromty == types.float32 and toty == types.float64:
                return builder.fpext(val, lty)
            elif fromty == types.float64 and toty == types.float32:
                return builder.fptrunc(val, lty)

        elif fromty in types.real_domain and toty in types.complex_domain:
            if fromty == types.float32:
                if toty == types.complex128:
                    real = self.cast(builder, val, fromty, types.float64)
                else:
                    real = val

            elif fromty == types.float64:
                if toty == types.complex64:
                    real = self.cast(builder, val, fromty, types.float32)
                else:
                    real = val

            if toty == types.complex128:
                imag = self.get_constant(types.float64, 0)
            elif toty == types.complex64:
                imag = self.get_constant(types.float32, 0)
            else:
                raise Exception("unreachable")

            cmplx = self.make_complex(toty)(self, builder)
            cmplx.real = real
            cmplx.imag = imag
            return cmplx._getvalue()

        elif fromty in types.integer_domain and toty in types.real_domain:
            lty = self.get_value_type(toty)
            if fromty in types.signed_domain:
                return builder.sitofp(val, lty)
            else:
                return builder.uitofp(val, lty)

        elif toty in types.integer_domain and fromty in types.real_domain:
            lty = self.get_value_type(toty)
            if toty in types.signed_domain:
                return builder.fptosi(val, lty)
            else:
                return builder.fptoui(val, lty)

        elif fromty in types.integer_domain and toty in types.complex_domain:
            cmplxcls, flty = builtins.get_complex_info(toty)
            cmpl = cmplxcls(self, builder)
            cmpl.real = self.cast(builder, val, fromty, flty)
            cmpl.imag = self.get_constant(flty, 0)
            return cmpl._getvalue()

        elif fromty in types.complex_domain and toty in types.complex_domain:
            srccls, srcty = builtins.get_complex_info(fromty)
            dstcls, dstty = builtins.get_complex_info(toty)

            src = srccls(self, builder, value=val)
            dst = dstcls(self, builder)
            dst.real = self.cast(builder, src.real, srcty, dstty)
            dst.imag = self.cast(builder, src.imag, srcty, dstty)
            return dst._getvalue()

        elif (isinstance(toty, types.UniTuple) and
                  isinstance(fromty, types.UniTuple) and
                      len(fromty) == len(toty)):
            olditems = cgutils.unpack_tuple(builder, val, len(fromty))
            items = [self.cast(builder, i, fromty.dtype, toty.dtype)
                     for i in olditems]
            tup = self.get_constant_undef(toty)
            for idx, val in enumerate(items):
                tup = builder.insert_value(tup, val, idx)
            return tup

        elif (isinstance(fromty, (types.UniTuple, types.Tuple)) and
                  isinstance(toty, (types.UniTuple, types.Tuple)) and
                      len(toty) == len(fromty)):

            olditems = cgutils.unpack_tuple(builder, val, len(fromty))
            items = [self.cast(builder, i, f, t)
                     for i, f, t in zip(olditems, fromty, toty)]
            tup = self.get_constant_undef(toty)
            for idx, val in enumerate(items):
                tup = builder.insert_value(tup, val, idx)
            return tup

        elif toty == types.boolean:
            return self.is_true(builder, fromty, val)

        elif fromty == types.boolean:
            # first promote to int32
            asint = builder.zext(val, Type.int())
            # then promote to number
            return self.cast(builder, asint, types.int32, toty)

        elif fromty == types.none and isinstance(toty, types.Optional):
            return self.make_optional_none(builder, toty.type)

        elif isinstance(toty, types.Optional):
            casted = self.cast(builder, val, fromty, toty.type)
            return self.make_optional_value(builder, toty.type, casted)

        elif isinstance(fromty, types.Optional):
            optty = self.make_optional(fromty)
            optval = optty(self, builder, value=val)
            validbit = cgutils.as_bool_bit(builder, optval.valid)
            with cgutils.if_unlikely(builder, builder.not_(validbit)):
                self.return_errcode(builder, errcode.NONE_TYPE_ERROR)

            return optval.data

        elif (isinstance(fromty, types.Array) and
                  isinstance(toty, types.Array)):
            # Type inference should have prevented illegal array casting.
            assert toty.layout == 'A'
            return val

        raise NotImplementedError("cast", val, fromty, toty)

    def make_optional(self, optionaltype):
        return optional.make_optional(optionaltype.type)

    def make_optional_none(self, builder, valtype):
        optcls = optional.make_optional(valtype)
        optval = optcls(self, builder)
        optval.valid = cgutils.false_bit
        return optval._getvalue()

    def make_optional_value(self, builder, valtype, value):
        optcls = optional.make_optional(valtype)
        optval = optcls(self, builder)
        optval.valid = cgutils.true_bit
        optval.data = value
        return optval._getvalue()

    def is_true(self, builder, typ, val):
        if typ in types.integer_domain:
            return builder.icmp(lc.ICMP_NE, val, Constant.null(val.type))
        elif typ in types.real_domain:
            return builder.fcmp(lc.FCMP_UNE, val, Constant.real(val.type, 0))
        elif typ in types.complex_domain:
            cmplx = self.make_complex(typ)(self, builder, val)
            real_istrue = self.is_true(builder, typ.underlying_float, cmplx.real)
            imag_istrue = self.is_true(builder, typ.underlying_float, cmplx.imag)
            return builder.or_(real_istrue, imag_istrue)
        raise NotImplementedError("is_true", val, typ)

    def call_function(self, builder, callee, resty, argtys, args, env=None):
        """
        Call the Numba-compiled *callee*, using the same calling
        convention as in get_function_type().
        """
        assert env is None
        retty = callee.args[0].type.pointee
        retvaltmp = cgutils.alloca_once(builder, retty)
        # initialize return value
        builder.store(lc.Constant.null(retty), retvaltmp)
        args = [self.get_value_as_argument(builder, ty, arg)
                for ty, arg in zip(argtys, args)]
        realargs = [retvaltmp] + list(args)
        code = builder.call(callee, realargs)
        status = self.get_return_status(builder, code)
        retval = builder.load(retvaltmp)
        out = self.get_returned_value(builder, resty, retval)
        return status, out

    def call_external_function(self, builder, callee, argtys, args):
        args = [self.get_value_as_argument(builder, ty, arg)
                for ty, arg in zip(argtys, args)]
        retval = builder.call(callee, args)
        return retval

    def get_return_status(self, builder, code):
        norm = builder.icmp(lc.ICMP_EQ, code, RETCODE_OK)
        none = builder.icmp(lc.ICMP_EQ, code, RETCODE_NONE)
        exc = builder.icmp(lc.ICMP_EQ, code, RETCODE_EXC)
        ok = builder.or_(norm, none)
        err = builder.not_(ok)

        status = Status(code=code, ok=ok, err=err, exc=exc, none=none)
        return status

    def call_function_pointer(self, builder, funcptr, signature, args, cconv=None):
        retty = self.get_value_type(signature.return_type)
        fnty = Type.function(retty, [a.type for a in args])
        fnptrty = Type.pointer(fnty)
        addr = self.get_constant(types.intp, funcptr)
        ptr = builder.inttoptr(addr, fnptrty)
        return builder.call(ptr, args, cconv=cconv)

    def call_class_method(self, builder, func, signature, args):
        api = self.get_python_api(builder)
        tys = signature.args
        retty = signature.return_type
        pyargs = [api.from_native_value(av, at) for av, at in zip(args, tys)]
        res = api.call_function_objargs(func, pyargs)

        # clean up
        api.decref(func)
        for obj in pyargs:
            api.decref(obj)

        with cgutils.ifthen(builder, cgutils.is_null(builder, res)):
            self.return_exc(builder)

        if retty == types.none:
            api.decref(res)
            return self.get_dummy_value()
        else:
            nativeresult = api.to_native_value(res, retty)
            api.decref(res)
            return nativeresult

    def print_string(self, builder, text):
        mod = builder.basic_block.function.module
        cstring = GENERIC_POINTER
        fnty = Type.function(Type.int(), [cstring])
        puts = mod.get_or_insert_function(fnty, "puts")
        return builder.call(puts, [text])

    def debug_print(self, builder, text):
        mod = cgutils.get_module(builder)
        cstr = self.insert_const_string(mod, str(text))
        self.print_string(builder, cstr)

    def get_struct_member_type(self, member_type):
        """
        Get the LLVM type for struct member of type *member_type*.
        """
        # get_struct_type() will:
        # - represent Records as pointers
        # - represent everything else as plain data
        if isinstance(member_type, types.Record):
            return self.get_value_type(member_type)
        else:
            return self.get_data_type(member_type)

    def get_struct_type(self, struct):
        """
        Get the LLVM struct type for the given Structure class *struct*.
        """
        fields = [self.get_struct_member_type(v) for _, v in struct._fields]
        return Type.struct(fields)

    def get_dummy_value(self):
        return Constant.null(self.get_dummy_type())

    def get_dummy_type(self):
        return GENERIC_POINTER

    def compile_internal(self, builder, impl, sig, args, locals={}):
        """Invoke compiler to implement a function for a nopython function
        """
        cache_key = (impl.__code__, sig)
        fndesc = self.cached_internal_func.get(cache_key)

        if fndesc is None:
            # Compile
            from numba import compiler

            codegen = self.jit_codegen()
            library = codegen.create_library(impl.__name__)
            flags = compiler.Flags()
            flags.set('no_compile')
            flags.set('no_cpython_wrapper')
            cres = compiler.compile_internal(self.typing_context, self,
                                             library,
                                             impl, sig.args,
                                             sig.return_type, flags,
                                             locals=locals)

            # Allow inlining the function inside callers.
            codegen.add_linking_library(cres.library)
            fndesc = cres.fndesc
            self.cached_internal_func[cache_key] = fndesc

        # Add call to the generated function
        llvm_mod = cgutils.get_module(builder)
        fn = self.declare_function(llvm_mod, fndesc)
        status, res = self.call_function(builder, fn, sig.return_type,
                                         sig.args, args)

        return res

    def get_executable(self, func, fndesc):
        raise NotImplementedError

    def get_python_api(self, builder):
        return PythonAPI(self, builder)

    def sentry_record_alignment(self, rectyp, attr):
        """
        Assumes offset starts from a properly aligned location
        """
        if self.strict_alignment:
            offset = rectyp.offset(attr)
            elemty = rectyp.typeof(attr)
            align = self.get_abi_sizeof(self.get_data_type(elemty))
            if offset % align:
                msg = "{rec}.{attr} of type {type} is not aligned".format(
                    rec=rectyp, attr=attr, type=elemty)
                raise TypeError(msg)

    def make_array(self, typ):
        return arrayobj.make_array(typ)

    def make_complex(self, typ):
        cls, _ = builtins.get_complex_info(typ)
        return cls

    def make_pair(self, first_type, second_type):
        """
        Create a heterogenous pair class parametered for the given types.
        """
        return builtins.make_pair(first_type, second_type)

    def make_constant_array(self, builder, typ, ary):
        assert typ.layout == 'C'                # assumed in typeinfer.py
        ary = numpy.ascontiguousarray(ary)
        flat = ary.flatten()

        # Handle data
        if self.is_struct_type(typ.dtype):
            values = [self.get_constant_struct(builder, typ.dtype, flat[i])
                      for i in range(flat.size)]
        else:
            values = [self.get_constant(typ.dtype, flat[i])
                      for i in range(flat.size)]

        lldtype = values[0].type
        consts = Constant.array(lldtype, values)
        data = cgutils.global_constant(builder, ".const.array.data", consts)

        # Handle shape
        llintp = self.get_value_type(types.intp)
        shapevals = [self.get_constant(types.intp, s) for s in ary.shape]
        cshape = Constant.array(llintp, shapevals)


        # Handle strides
        stridevals = [self.get_constant(types.intp, s) for s in ary.strides]
        cstrides = Constant.array(llintp, stridevals)

        # Create array structure
        cary = self.make_array(typ)(self, builder)
        cary.data = builder.bitcast(data, cary.data.type)
        cary.shape = cshape
        cary.strides = cstrides
        return cary._getvalue()

    def get_abi_sizeof(self, ty):
        """
        Get the ABI size of LLVM type *ty*.
        """
        if isinstance(ty, llvmir.Type):
            return ty.get_abi_size(self.target_data)
        # XXX this one unused?
        return self.target_data.get_abi_size(ty)

    def post_lowering(self, func):
        """Run target specific post-lowering transformation here.
        """
        pass

    def create_module(self, name):
        """Create a LLVM module
        """
        return lc.Module.new(name)


class _wrap_impl(object):
    def __init__(self, imp, context, sig):
        self._imp = imp
        self._context = context
        self._sig = sig

    def __call__(self, builder, args):
        return self._imp(self._context, builder, self._sig, args)

    def __getattr__(self, item):
        return getattr(self._imp, item)

    def __repr__(self):
        return "<wrapped %s>" % self._imp

