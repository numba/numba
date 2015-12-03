from __future__ import print_function

from collections import namedtuple, defaultdict
import copy
import sys
from types import MethodType

import numpy

from llvmlite import ir as llvmir
import llvmlite.llvmpy.core as lc
from llvmlite.llvmpy.core import Type, Constant, LLVMException
import llvmlite.binding as ll

from numba import types, utils, cgutils, typing
from numba import _dynfunc, _helperlib
from numba.pythonapi import PythonAPI
from numba.targets.imputils import (user_function, user_generator,
                                    builtin_registry, impl_attribute,
                                    impl_ret_borrowed)
from . import (
    arrayobj, arraymath, builtins, iterators, rangeobj, optional, slicing,
    tupleobj)
from numba import datamodel

try:
    from . import npdatetime
except NotImplementedError:
    pass


GENERIC_POINTER = Type.pointer(Type.int(8))
PYOBJECT = GENERIC_POINTER
void_ptr = GENERIC_POINTER

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
    types.slice3_type: slicing.Slice,
}


class Overloads(object):
    def __init__(self):
        # A list of (signature, implementation)
        self.versions = []

    def find(self, sig):
        for ver_sig, impl in self.versions:
            if ver_sig == sig:
                return impl

            # As generic type
            if self._match_arglist(ver_sig.args, sig.args):
                return impl

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

    def append(self, impl, sig):
        self.versions.append((sig, impl))


@utils.runonce
def _load_global_helpers():
    """
    Execute once to install special symbols into the LLVM symbol table.
    """
    # This is Py_None's real C name
    ll.add_symbol("_Py_NoneStruct", id(None))

    # Add C helper functions
    for c_helpers in (_helperlib.c_helpers, _dynfunc.c_helpers):
        for py_name in c_helpers:
            c_name = "numba_" + py_name
            c_address = c_helpers[py_name]
            ll.add_symbol(c_name, c_address)

    # Add all built-in exception classes
    for obj in utils.builtins.__dict__.values():
        if isinstance(obj, type) and issubclass(obj, BaseException):
            ll.add_symbol("PyExc_%s" % (obj.__name__), id(obj))


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

    # Bound checking
    enable_boundcheck = False

    # NRT
    enable_nrt = False

    # PYCC
    aot_mode = False

    # Error model for various operations (only FP exceptions currently)
    error_model = None

    def __init__(self, typing_context):
        _load_global_helpers()
        self.address_size = utils.MACHINE_BITS
        self.typing_context = typing_context

        self.defns = defaultdict(Overloads)
        self.attrs = defaultdict(Overloads)
        self.generators = {}
        self.special_ops = {}

        self.install_registry(builtin_registry)

        self.cached_internal_func = {}

        self.data_model_manager = datamodel.default_manager

        # Initialize
        self.init()

    def init(self):
        """
        For subclasses to add initializer
        """
        pass

    def get_arg_packer(self, fe_args):
        return datamodel.ArgPacker(self.data_model_manager, fe_args)

    @property
    def target_data(self):
        raise NotImplementedError

    def subtarget(self, **kws):
        obj = copy.copy(self)  # shallow copy
        for k, v in kws.items():
            if not hasattr(obj, k):
                raise NameError("unknown option {0!r}".format(k))
            setattr(obj, k, v)
        if obj.codegen() is not self.codegen():
            # We can't share functions accross different codegens
            obj.cached_internal_func = {}
        return obj

    def install_registry(self, registry):
        """
        Install a *registry* (a imputils.Registry instance) of function
        and attribute implementations.
        """
        self.insert_func_defn(registry.functions)
        self.insert_attr_defn(registry.attributes)

    def insert_func_defn(self, defns):
        for impl, func_sigs in defns:
            for func, sig in func_sigs:
                self.defns[func].append(impl, sig)

    def insert_attr_defn(self, defns):
        for impl in defns:
            self.attrs[impl.attr].append(impl, impl.signature)

    def insert_user_function(self, func, fndesc, libs=()):
        impl = user_function(fndesc, libs)
        self.defns[func].append(impl, impl.signature)

    def add_user_function(self, func, fndesc, libs=()):
        if func not in self.defns:
            msg = "{func} is not a registered user function"
            raise KeyError(msg.format(func=func))
        impl = user_function(fndesc, libs)
        self.defns[func].append(impl, impl.signature)

    def insert_generator(self, genty, gendesc, libs=()):
        assert isinstance(genty, types.Generator)
        impl = user_generator(gendesc, libs)
        self.generators[genty] = gendesc, impl

    def remove_user_function(self, func):
        """
        Remove user function *func*.
        KeyError is raised if the function isn't known to us.
        """
        del self.defns[func]

    def get_external_function_type(self, fndesc):
        argtypes = [self.get_argument_type(aty)
                    for aty in fndesc.argtypes]
        # don't wrap in pointer
        restype = self.get_argument_type(fndesc.restype)
        fnty = Type.function(restype, argtypes)
        return fnty

    def declare_function(self, module, fndesc):
        fnty = self.call_conv.get_function_type(fndesc.restype, fndesc.argtypes)
        fn = module.get_or_insert_function(fnty, name=fndesc.mangled_name)
        assert fn.is_declaration
        self.call_conv.decorate_function(fn, fndesc.args, fndesc.argtypes)
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
        """
        Insert constant *string* (a str object) into module *mod*.
        """
        stringtype = GENERIC_POINTER
        name = ".const.%s" % string
        text = cgutils.make_bytearray(string.encode("utf-8") + b"\x00")
        gv = self.insert_unique_const(mod, name, text)
        return Constant.bitcast(gv, stringtype)

    def insert_unique_const(self, mod, name, val):
        """
        Insert a unique internal constant named *name*, with LLVM value
        *val*, into module *mod*.
        """
        gv = mod.get_global(name)
        if gv is not None:
            return gv
        else:
            return cgutils.global_constant(mod, name, val)

    def get_argument_type(self, ty):
        return self.data_model_manager[ty].get_argument_type()

    def get_return_type(self, ty):
        return self.data_model_manager[ty].get_return_type()

    def get_data_type(self, ty):
        """
        Get a LLVM data representation of the Numba type *ty* that is safe
        for storage.  Record data are stored as byte array.

        The return value is a llvmlite.ir.Type object, or None if the type
        is an opaque pointer (???).
        """
        return self.data_model_manager[ty].get_data_type()

    def get_value_type(self, ty):
        return self.data_model_manager[ty].get_value_type()

    def pack_value(self, builder, ty, value, ptr, align=None):
        """
        Pack value into the array storage at *ptr*.
        If *align* is given, it is the guaranteed alignment for *ptr*
        (by default, the standard ABI alignment).
        """
        dataval = self.data_model_manager[ty].as_data(builder, value)
        builder.store(dataval, ptr, align=align)

    def unpack_value(self, builder, ty, ptr, align=None):
        """
        Unpack value from the array storage at *ptr*.
        If *align* is given, it is the guaranteed alignment for *ptr*
        (by default, the standard ABI alignment).
        """
        dm = self.data_model_manager[ty]
        return dm.load_from_data_pointer(builder, ptr, align)

    def is_struct_type(self, ty):
        return isinstance(self.data_model_manager[ty], datamodel.CompositeModel)

    def get_constant_generic(self, builder, ty, val):
        """
        Return a LLVM constant representing value *val* of Numba type *ty*.
        """
        if isinstance(ty, types.ExternalFunctionPointer):
            ptrty = self.get_function_pointer_type(ty)
            ptrval = ty.get_pointer(val)
            return builder.inttoptr(self.get_constant(types.intp, ptrval),
                                    ptrty)

        elif isinstance(ty, types.Array):
            return self.make_constant_array(builder, ty, val)

        elif isinstance(ty, types.Dummy):
            return self.get_dummy_value()

        elif self.is_struct_type(ty):
            struct = self.get_constant_struct(builder, ty, val)
            if isinstance(ty, types.Record):
                ptrty = self.data_model_manager[ty].get_data_type()
                ptr = cgutils.alloca_once(builder, ptrty)
                builder.store(struct, ptr)
                return ptr
            return struct

        else:
            return self.get_constant(ty, val)

    def get_constant_struct(self, builder, ty, val):
        assert self.is_struct_type(ty)

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

        elif isinstance(ty, (types.Tuple, types.NamedTuple)):
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

        elif isinstance(ty, (types.NPDatetime, types.NPTimedelta)):
            return Constant.real(lty, val.astype(numpy.int64))

        elif isinstance(ty, (types.UniTuple, types.NamedUniTuple)):
            consts = [self.get_constant(ty.dtype, v) for v in val]
            return Constant.array(consts[0].type, consts)

        raise NotImplementedError("cannot lower constant of type '%s'" % (ty,))

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
                align = None if typ.aligned else 1
                self.pack_value(builder, elemty, val, dptr, align=align)

            return _wrap_impl(imp, self, sig)

    def get_function(self, fn, sig):
        """
        Return the implementation of function *fn* for signature *sig*.
        The return value is a callable with the signature (builder, args).
        """
        if isinstance(fn, (types.Function)):
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
            pass
        if isinstance(fn, types.Type):
            # It's a type instance => try to find a definition for the type class
            return self.get_function(type(fn), sig)
        raise NotImplementedError("No definition for lowering %s%s" % (key, sig))

    def get_generator_desc(self, genty):
        """
        """
        return self.generators[genty][0]

    def get_generator_impl(self, genty):
        """
        """
        return self.generators[genty][1]

    def get_bound_function(self, builder, obj, ty):
        return obj

    def get_attribute(self, val, typ, attr):
        if isinstance(typ, types.Module):
            # Implement getattr for module-level globals.
            # We are treating them as constants.
            # XXX We shouldn't have to retype this
            attrty = self.typing_context.resolve_module_constants(typ, attr)
            if attrty is not None and not isinstance(attrty, types.Dummy):
                pyval = getattr(typ.pymod, attr)
                llval = self.get_constant(attrty, pyval)
                @impl_attribute(typ, attr, attrty)
                def imp(context, builder, typ, val):
                    return impl_ret_borrowed(context, builder, attrty, llval)
                return imp
            # No implementation required for dummies (functions, modules...),
            # which are dealt with later
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
        return self.data_model_manager[ty].from_argument(builder, val)

    def get_returned_value(self, builder, ty, val):
        """
        Return value representation to local value representation
        """
        return self.data_model_manager[ty].from_return(builder, val)

    def get_return_value(self, builder, ty, val):
        """
        Local value representation to return type representation
        """
        return self.data_model_manager[ty].as_return(builder, val)

    def get_value_as_argument(self, builder, ty, val):
        """Prepare local value representation as argument type representation
        """
        return self.data_model_manager[ty].as_argument(builder, val)

    def get_value_as_data(self, builder, ty, val):
        return self.data_model_manager[ty].as_data(builder, val)

    def get_data_as_value(self, builder, ty, val):
        return self.data_model_manager[ty].from_data(builder, val)

    def pair_first(self, builder, val, ty):
        """
        Extract the first element of a heterogenous pair.
        """
        paircls = self.make_pair(ty.first_type, ty.second_type)
        pair = paircls(self, builder, value=val)
        return pair.first

    def pair_second(self, builder, val, ty):
        """
        Extract the second element of a heterogenous pair.
        """
        paircls = self.make_pair(ty.first_type, ty.second_type)
        pair = paircls(self, builder, value=val)
        return pair.second

    def cast(self, builder, val, fromty, toty):
        if fromty == toty or toty == types.Any or isinstance(toty, types.Kind):
            return val

        elif isinstance(fromty, types.Integer) and isinstance(toty, types.Integer):
            if toty.bitwidth == fromty.bitwidth:
                # Just a change of signedness
                return val
            elif toty.bitwidth < fromty.bitwidth:
                # Downcast
                return builder.trunc(val, self.get_value_type(toty))
            elif fromty.signed:
                # Signed upcast
                return builder.sext(val, self.get_value_type(toty))
            else:
                # Unsigned upcast
                return builder.zext(val, self.get_value_type(toty))

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

        elif (isinstance(fromty, (types.UniTuple, types.Tuple)) and
              isinstance(toty, (types.UniTuple, types.Tuple)) and
              len(toty) == len(fromty)):
            olditems = cgutils.unpack_tuple(builder, val, len(fromty))
            items = [self.cast(builder, i, f, t)
                     for i, f, t in zip(olditems, fromty, toty)]
            return cgutils.make_anonymous_struct(builder, items)

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
                msg = "expected %s, got None" % (fromty.type,)
                self.call_conv.return_user_exc(builder, TypeError, (msg,))

            return optval.data

        elif (isinstance(fromty, types.Array) and
              isinstance(toty, types.Array)):
            # Type inference should have prevented illegal array casting.
            assert toty.layout == 'A'
            return val

        elif (isinstance(fromty, types.List) and
              isinstance(toty, types.List)):
            # Casting from non-reflected to reflected
            assert fromty.dtype == toty.dtype
            return val

        elif (isinstance(fromty, types.RangeType) and
              isinstance(toty, types.RangeType)):
            olditems = cgutils.unpack_tuple(builder, val, 3)
            items = [self.cast(builder, v, fromty.dtype, toty.dtype)
                     for v in olditems]
            return cgutils.make_anonymous_struct(builder, items)

        elif fromty in types.integer_domain and toty == types.voidptr:
            return builder.inttoptr(val, self.get_value_type(toty))

        raise NotImplementedError("cast", val, fromty, toty)

    def generic_compare(self, builder, key, argtypes, args):
        """
        Compare the given LLVM values of the given Numba types using
        the comparison *key* (e.g. '==').  The values are first cast to
        a common safe conversion type.
        """
        at, bt = argtypes
        av, bv = args
        ty = self.typing_context.unify_types(at, bt)
        cav = self.cast(builder, av, at, ty)
        cbv = self.cast(builder, bv, bt, ty)
        cmpsig = typing.signature(types.boolean, ty, ty)
        cmpfunc = self.get_function(key, cmpsig)
        return cmpfunc(builder, (cav, cbv))

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
        """
        Return the truth value of a value of the given Numba type.
        """
        impl = self.get_function(bool, typing.signature(types.boolean, typ))
        return impl(builder, (val,))

    def get_c_value(self, builder, typ, name, dllimport=False):
        """
        Get a global value through its C-accessible *name*, with the given
        LLVM type.
        If *dllimport* is true, the symbol will be marked as imported
        from a DLL (necessary for AOT compilation under Windows).
        """
        module = builder.function.module
        try:
            gv = module.get_global_variable_named(name)
        except LLVMException:
            gv = module.add_global_variable(typ, name)
            if dllimport and self.aot_mode and sys.platform == 'win32':
                gv.storage_class = "dllimport"
        return gv

    def call_external_function(self, builder, callee, argtys, args):
        args = [self.get_value_as_argument(builder, ty, arg)
                for ty, arg in zip(argtys, args)]
        retval = builder.call(callee, args)
        return retval

    def get_function_pointer_type(self, typ):
        return self.data_model_manager[typ].get_data_type()

    def call_function_pointer(self, builder, funcptr, args, cconv=None):
        return builder.call(funcptr, args, cconv=cconv)

    def print_string(self, builder, text):
        mod = builder.basic_block.function.module
        cstring = GENERIC_POINTER
        fnty = Type.function(Type.int(), [cstring])
        puts = mod.get_or_insert_function(fnty, "puts")
        return builder.call(puts, [text])

    def debug_print(self, builder, text):
        mod = builder.module
        cstr = self.insert_const_string(mod, str(text))
        self.print_string(builder, cstr)

    def get_struct_type(self, struct):
        """
        Get the LLVM struct type for the given Structure class *struct*.
        """
        fields = [self.get_value_type(v) for _, v in struct._fields]
        return Type.struct(fields)

    def get_dummy_value(self):
        return Constant.null(self.get_dummy_type())

    def get_dummy_type(self):
        return GENERIC_POINTER

    def compile_only_no_cache(self, builder, impl, sig, locals={}):
        """Invoke the compiler to compile a function to be used inside a
        nopython function, but without generating code to call that
        function.
        """
        # Compile
        from numba import compiler

        codegen = self.codegen()
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
        return cres

    def compile_subroutine(self, builder, impl, sig, locals={}):
        """
        Compile the function *impl* for the given *sig* (in nopython mode).
        Return a placeholder object that's callable from another Numba
        function.
        """
        cache_key = (impl.__code__, sig)
        if impl.__closure__:
            # XXX This obviously won't work if a cell's value is
            # unhashable.
            cache_key += tuple(c.cell_contents for c in impl.__closure__)
        ty = self.cached_internal_func.get(cache_key)
        if ty is None:
            cres = self.compile_only_no_cache(builder, impl, sig,
                                              locals=locals)
            ty = types.NumbaFunction(cres.fndesc, sig)
            self.cached_internal_func[cache_key] = ty
        return ty

    def compile_internal(self, builder, impl, sig, args, locals={}):
        """
        Like compile_subroutine(), but also call the function with the given
        *args*.
        """
        ty = self.compile_subroutine(builder, impl, sig, locals)
        return self.call_internal(builder, ty.fndesc, sig, args)

    def call_internal(self, builder, fndesc, sig, args):
        """Given the function descriptor of an internally compiled function,
        emit a call to that function with the given arguments.
        """
        # Add call to the generated function
        llvm_mod = builder.module
        fn = self.declare_function(llvm_mod, fndesc)
        status, res = self.call_conv.call_function(builder, fn, sig.return_type,
                                                   sig.args, args)

        with cgutils.if_unlikely(builder, status.is_error):
            self.call_conv.return_status_propagate(builder, status)
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
            align = self.get_abi_alignment(self.get_data_type(elemty))
            if offset % align:
                msg = "{rec}.{attr} of type {type} is not aligned".format(
                    rec=rectyp, attr=attr, type=elemty)
                raise TypeError(msg)

    def make_array(self, typ):
        return arrayobj.make_array(typ)

    def populate_array(self, arr, **kwargs):
        """
        Populate array structure.
        """
        return arrayobj.populate_array(arr, **kwargs)

    def make_complex(self, typ):
        cls, _ = builtins.get_complex_info(typ)
        return cls

    def make_pair(self, first_type, second_type):
        """
        Create a heterogenous pair class parametered for the given types.
        """
        return builtins.make_pair(first_type, second_type)

    def make_tuple(self, builder, typ, values):
        """
        Create a tuple of the given *typ* containing the *values*.
        """
        tup = self.get_constant_undef(typ)
        for i, val in enumerate(values):
            tup = builder.insert_value(tup, val, i)
        return tup

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

        rt_addr = self.get_constant(types.uintp, id(ary)).inttoptr(
            self.get_value_type(types.pyobject))

        intp_itemsize = self.get_constant(types.intp, ary.dtype.itemsize)
        self.populate_array(cary,
                            data=builder.bitcast(data, cary.data.type),
                            shape=cshape,
                            strides=cstrides,
                            itemsize=intp_itemsize,
                            parent=rt_addr,
                            meminfo=None)

        return cary._getvalue()

    def get_abi_sizeof(self, ty):
        """
        Get the ABI size of LLVM type *ty*.
        """
        if isinstance(ty, llvmir.Type):
            return ty.get_abi_size(self.target_data)
        # XXX this one unused?
        return self.target_data.get_abi_size(ty)

    def get_abi_alignment(self, ty):
        """
        Get the ABI alignment of LLVM type *ty*.
        """
        assert isinstance(ty, llvmir.Type), "Expected LLVM type"
        return ty.get_abi_alignment(self.target_data)

    def post_lowering(self, mod, library):
        """Run target specific post-lowering transformation here.
        """

    def create_module(self, name):
        """Create a LLVM module
        """
        return lc.Module.new(name)

    def nrt_meminfo_alloc(self, builder, size):
        """
        Allocate a new MemInfo with a data payload of `size` bytes.

        A pointer to the MemInfo is returned.
        """
        if not self.enable_nrt:
            raise Exception("Require NRT")
        mod = builder.module
        fnty = llvmir.FunctionType(void_ptr,
                                   [self.get_value_type(types.intp)])
        fn = mod.get_or_insert_function(fnty, name="NRT_MemInfo_alloc_safe")
        fn.return_value.add_attribute("noalias")
        return builder.call(fn, [size])

    def nrt_meminfo_alloc_aligned(self, builder, size, align):
        """
        Allocate a new MemInfo with an aligned data payload of `size` bytes.
        The data pointer is aligned to `align` bytes.  `align` can be either
        a Python int or a LLVM uint32 value.

        A pointer to the MemInfo is returned.
        """
        if not self.enable_nrt:
            raise Exception("Require NRT")
        mod = builder.module
        intp = self.get_value_type(types.intp)
        u32 = self.get_value_type(types.uint32)
        fnty = llvmir.FunctionType(void_ptr, [intp, u32])
        fn = mod.get_or_insert_function(fnty,
                                        name="NRT_MemInfo_alloc_safe_aligned")
        fn.return_value.add_attribute("noalias")
        if isinstance(align, int):
            align = self.get_constant(types.uint32, align)
        else:
            assert align.type == u32, "align must be a uint32"
        return builder.call(fn, [size, align])

    def nrt_meminfo_varsize_alloc(self, builder, size):
        """
        Allocate a MemInfo pointing to a variable-sized data area.  The area
        is separately allocated (i.e. two allocations are made) so that
        re-allocating it doesn't change the MemInfo's address.

        A pointer to the MemInfo is returned.
        """
        if not self.enable_nrt:
            raise Exception("Require NRT")
        mod = builder.module
        fnty = llvmir.FunctionType(void_ptr,
                                   [self.get_value_type(types.intp)])
        fn = mod.get_or_insert_function(fnty, name="NRT_MemInfo_varsize_alloc")
        fn.return_value.add_attribute("noalias")
        return builder.call(fn, [size])

    def nrt_meminfo_varsize_realloc(self, builder, meminfo, size):
        """
        Reallocate a data area allocated by nrt_meminfo_varsize_alloc().
        The new data pointer is returned, for convenience.
        """
        if not self.enable_nrt:
            raise Exception("Require NRT")
        mod = builder.module
        fnty = llvmir.FunctionType(void_ptr,
                                   [void_ptr, self.get_value_type(types.intp)])
        fn = mod.get_or_insert_function(fnty, name="NRT_MemInfo_varsize_realloc")
        fn.return_value.add_attribute("noalias")
        return builder.call(fn, [meminfo, size])

    def nrt_meminfo_data(self, builder, meminfo):
        """
        Given a MemInfo pointer, return a pointer to the allocated data
        managed by it.  This works for MemInfos allocated with all the
        above methods.
        """
        if not self.enable_nrt:
            raise Exception("Require NRT")
        from numba.runtime.atomicops import meminfo_data_ty

        mod = builder.module
        fn = mod.get_or_insert_function(meminfo_data_ty,
                                        name="NRT_MemInfo_data_fast")
        return builder.call(fn, [meminfo])

    def _call_nrt_incref_decref(self, builder, root_type, typ, value, funcname):
        if not self.enable_nrt:
            raise Exception("Require NRT")
        from numba.runtime.atomicops import incref_decref_ty

        data_model = self.data_model_manager[typ]

        members = data_model.traverse(builder, value)
        for mt, mv in members:
            self._call_nrt_incref_decref(builder, root_type, mt, mv, funcname)

        try:
            meminfo = data_model.get_nrt_meminfo(builder, value)
        except NotImplementedError as e:
            raise NotImplementedError("%s: %s" % (root_type, str(e)))
        if meminfo:
            mod = builder.module
            fn = mod.get_or_insert_function(incref_decref_ty, name=funcname)
            # XXX "nonnull" causes a crash in test_dyn_array: can this
            # function be called with a NULL pointer?
            fn.args[0].add_attribute("noalias")
            fn.args[0].add_attribute("nocapture")
            builder.call(fn, [meminfo])

    def nrt_incref(self, builder, typ, value):
        """
        Recursively incref the given *value* and its members.
        """
        self._call_nrt_incref_decref(builder, typ, typ, value, "NRT_incref")

    def nrt_decref(self, builder, typ, value):
        """
        Recursively decref the given *value* and its members.
        """
        self._call_nrt_incref_decref(builder, typ, typ, value, "NRT_decref")


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
