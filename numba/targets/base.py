from __future__ import print_function
from collections import namedtuple, defaultdict
import copy
from types import MethodType

import numpy

from llvmlite import ir as llvmir
import llvmlite.llvmpy.core as lc
from llvmlite.llvmpy.core import Type, Constant, LLVMException
import llvmlite.binding as ll

import numba
from numba import types, utils, cgutils, typing, numpy_support
from numba import _dynfunc, _helperlib
from numba.pythonapi import PythonAPI
from numba.targets.imputils import (user_function, user_generator,
                                    python_attr_impl,
                                    builtin_registry, impl_attribute,
                                    struct_registry, type_registry)
from . import arrayobj, builtins, iterators, rangeobj, optional
from numba import datamodel

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
    ll.add_symbol("Py_None", id(None))

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

    def insert_class(self, cls, attrs):
        clsty = types.Object(cls)
        for name, vtype in utils.iteritems(attrs):
            impl = python_attr_impl(clsty, name, vtype)
            self.attrs[impl.attr].append(impl, impl.signature)

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
        try:
            fac = type_registry.match(ty)
        except KeyError:
            pass
        else:
            return fac(self, ty)

        return self.data_model_manager[ty].get_data_type()

    def get_value_type(self, ty):
        return self.data_model_manager[ty].get_value_type()

    def pack_value(self, builder, ty, value, ptr):
        """Pack data for array storage
        """
        dataval = self.data_model_manager[ty].as_data(builder, value)
        builder.store(dataval, ptr)

    def unpack_value(self, builder, ty, ptr):
        """Unpack data from array storage
        """
        dm = self.data_model_manager[ty]
        return dm.load_from_data_pointer(builder, ptr)

    def is_struct_type(self, ty):
        return isinstance(self.data_model_manager[ty], datamodel.CompositeModel)

    def get_constant_generic(self, builder, ty, val):
        """
        Return a LLVM constant representing value *val* of Numba type *ty*.
        """
        if self.is_struct_type(ty):
            struct = self.get_constant_struct(builder, ty, val)
            if isinstance(ty, types.Record):
                ptrty = self.data_model_manager[ty].get_data_type()
                ptr = cgutils.alloca_once(builder, ptrty)
                builder.store(struct, ptr)
                return ptr
            return struct

        elif isinstance(ty, types.ExternalFunctionPointer):
            ptrty = self.get_function_pointer_type(ty)
            ptrval = ty.get_pointer(val)
            return builder.inttoptr(self.get_constant(types.intp, ptrval),
                                    ptrty)

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
                self.pack_value(builder, elemty, val, dptr)

            return _wrap_impl(imp, self, sig)

    def get_function(self, fn, sig):
        """
        Return the implementation of function *fn* for signature *sig*.
        The return value is a callable with the signature (builder, args).
        """
        if isinstance(fn, (types.NumberClass, types.Function)):
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
        if isinstance(typ, types.Record):
            # Implement get attribute for records
            self.sentry_record_alignment(typ, attr)
            offset = typ.offset(attr)
            elemty = typ.typeof(attr)

            if isinstance(elemty, types.NestedArray):
                # Inside a structured type only the array data is stored, so we
                # create an array structure to point to that data.
                aryty = arrayobj.make_array(elemty)
                @impl_attribute(typ, attr, elemty)
                def imp(context, builder, typ, val):
                    ary = aryty(context, builder)
                    dtype = elemty.dtype
                    newshape = [self.get_constant(types.intp, s) for s in
                                elemty.shape]
                    newstrides = [self.get_constant(types.intp, s) for s in
                                  elemty.strides]
                    newdata = cgutils.get_record_member(builder, val, offset,
                                                    self.get_data_type(dtype))
                    arrayobj.populate_array(
                        ary,
                        data=newdata,
                        shape=cgutils.pack_array(builder, newshape),
                        strides=cgutils.pack_array(builder, newstrides),
                        itemsize=context.get_constant(types.intp, elemty.size),
                        meminfo=None,
                        parent=None,
                    )

                    return ary._getvalue()
            else:
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
            if attrty is not None and not isinstance(attrty, types.Dummy):
                pyval = getattr(typ.pymod, attr)
                llval = self.get_constant(attrty, pyval)
                @impl_attribute(typ, attr, attrty)
                def imp(context, builder, typ, val):
                    return llval
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
                msg = "expected %s, got None" % (fromty.type,)
                self.call_conv.return_user_exc(builder, TypeError, (msg,))

            return optval.data

        elif (isinstance(fromty, types.Array) and
                  isinstance(toty, types.Array)):
            # Type inference should have prevented illegal array casting.
            assert toty.layout == 'A'
            return val

        elif fromty in types.integer_domain and toty == types.voidptr:
            return builder.inttoptr(val, self.get_value_type(toty))

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

    def get_c_value(self, builder, typ, name):
        """
        Get a global value through its C-accessible *name*, with the given
        LLVM type.
        """
        module = builder.function.module
        try:
            gv = module.get_global_variable_named(name)
        except LLVMException:
            gv = module.add_global_variable(typ, name)
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
            self.call_conv.return_exc(builder)

        if retty == types.none:
            api.decref(res)
            return self.get_dummy_value()
        else:
            native = api.to_native_value(res, retty)
            assert native.cleanup is None
            api.decref(res)
            return native.value

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
        return cres

    def compile_internal(self, builder, impl, sig, args, locals={}):
        """Invoke compiler to implement a function for a nopython function
        """
        cache_key = (impl.__code__, sig)
        if impl.__closure__:
            # XXX This obviously won't work if a cell's value is
            # unhashable.
            cache_key += tuple(c.cell_contents for c in impl.__closure__)
        fndesc = self.cached_internal_func.get(cache_key)

        if fndesc is None:
            cres = self.compile_only_no_cache(builder, impl, sig,
                                              locals=locals)
            fndesc = cres.fndesc
            self.cached_internal_func[cache_key] = fndesc

        return self.call_internal(builder, fndesc, sig, args)

    def call_internal(self, builder, fndesc, sig, args):
        """Given the function descriptor of an internally compiled function,
        emit a call to that function with the given arguments.
        """
        # Add call to the generated function
        llvm_mod = cgutils.get_module(builder)
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

    def post_lowering(self, func):
        """Run target specific post-lowering transformation here.
        """
        pass

    def create_module(self, name):
        """Create a LLVM module
        """
        return lc.Module.new(name)

    def nrt_meminfo_alloc(self, builder, size):
        if not self.enable_nrt:
            raise Exception("Require NRT")
        mod = cgutils.get_module(builder)
        fnty = llvmir.FunctionType(llvmir.IntType(8).as_pointer(),
            [self.get_value_type(types.intp)])
        fn = mod.get_or_insert_function(fnty, name="NRT_MemInfo_alloc_safe")
        return builder.call(fn, [size])

    def nrt_meminfo_data(self, builder, meminfo):
        if not self.enable_nrt:
            raise Exception("Require NRT")
        mod = cgutils.get_module(builder)
        voidptr = llvmir.IntType(8).as_pointer()
        fnty = llvmir.FunctionType(voidptr, [voidptr])
        fn = mod.get_or_insert_function(fnty, name="NRT_MemInfo_data")
        return builder.call(fn, [meminfo])

    def get_nrt_meminfo(self, builder, typ, value):
        return self.data_model_manager[typ].get_nrt_meminfo(builder, value)

    def nrt_incref(self, builder, typ, value):
        if not self.enable_nrt:
            raise Exception("Require NRT")

        members = self.data_model_manager[typ].traverse(builder, value)
        for mt, mv in members:
            self.nrt_incref(builder, mt, mv)

        meminfo = self.get_nrt_meminfo(builder, typ, value)
        if meminfo:
            mod = cgutils.get_module(builder)
            fnty = llvmir.FunctionType(llvmir.VoidType(),
                [llvmir.IntType(8).as_pointer()])
            fn = mod.get_or_insert_function(fnty, name="NRT_incref")

            builder.call(fn, [meminfo])

    def nrt_decref(self, builder, typ, value):
        if not self.enable_nrt:
            raise Exception("Require NRT")

        members = self.data_model_manager[typ].traverse(builder, value)
        for mt, mv in members:
            self.nrt_decref(builder, mt, mv)

        meminfo = self.get_nrt_meminfo(builder, typ, value)
        if meminfo:
            mod = cgutils.get_module(builder)
            fnty = llvmir.FunctionType(llvmir.VoidType(),
                [llvmir.IntType(8).as_pointer()])
            fn = mod.get_or_insert_function(fnty, name="NRT_decref")
            builder.call(fn, [meminfo])


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

