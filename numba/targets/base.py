from __future__ import print_function

from collections import namedtuple, defaultdict
import copy
import os
import sys
from itertools import permutations, takewhile

import numpy as np

from llvmlite import ir as llvmir
import llvmlite.llvmpy.core as lc
from llvmlite.llvmpy.core import Type, Constant, LLVMException
import llvmlite.binding as ll

from numba import types, utils, cgutils, typing
from numba import _dynfunc, _helperlib
from numba.pythonapi import PythonAPI
from . import arrayobj, builtins, imputils
from .imputils import (user_function, user_generator,
                       builtin_registry, impl_ret_borrowed,
                       RegistryLoader)
from numba import datamodel


GENERIC_POINTER = Type.pointer(Type.int(8))
PYOBJECT = GENERIC_POINTER
void_ptr = GENERIC_POINTER


class OverloadSelector(object):
    """
    An object matching an actual signature against a registry of formal
    signatures and choosing the best candidate, if any.

    In the current implementation:
    - a "signature" is a tuple of type classes or type instances
    - the "best candidate" is the most specific match
    """

    def __init__(self):
        # A list of (formal args tuple, value)
        self.versions = []
        self._cache = {}

    def find(self, sig):
        out = self._cache.get(sig)
        if out is None:
            out = self._find(sig)
            self._cache[sig] = out
        return out

    def _find(self, sig):
        candidates = self._select_compatible(sig)
        if candidates:
            return candidates[self._best_signature(candidates)]
        else:
            raise NotImplementedError(self, sig)

    def _select_compatible(self, sig):
        """
        Select all compatible signatures and their implementation.
        """
        out = {}
        for ver_sig, impl in self.versions:
            if self._match_arglist(ver_sig, sig):
                out[ver_sig] = impl
        return out

    def _best_signature(self, candidates):
        """
        Returns the best signature out of the candidates
        """
        ordered, genericity = self._sort_signatures(candidates)
        # check for ambiguous signatures
        if len(ordered) > 1:
            firstscore = genericity[ordered[0]]
            same = list(takewhile(lambda x: genericity[x] == firstscore,
                                  ordered))
            if len(same) > 1:
                msg = ["{n} ambiguous signatures".format(n=len(same))]
                for sig in same:
                    msg += ["{0} => {1}".format(sig, candidates[sig])]
                raise TypeError('\n'.join(msg))
        return ordered[0]

    def _sort_signatures(self, candidates):
        """
        Sort signatures in ascending level of genericity.

        Returns a 2-tuple:

            * ordered list of signatures
            * dictionary containing genericity scores
        """
        # score by genericity
        genericity = defaultdict(int)
        for this, other in permutations(candidates.keys(), r=2):
            matched = self._match_arglist(formal_args=this, actual_args=other)
            if matched:
                # genericity score +1 for every another compatible signature
                genericity[this] += 1
        # order candidates in ascending level of genericity
        ordered = sorted(candidates.keys(), key=lambda x: genericity[x])
        return ordered, genericity

    def _match_arglist(self, formal_args, actual_args):
        """
        Returns True if the the signature is "matching".
        A formal signature is "matching" if the actual signature matches exactly
        or if the formal signature is a compatible generic signature.
        """
        # normalize VarArg
        if formal_args and isinstance(formal_args[-1], types.VarArg):
            ndiff = len(actual_args) - len(formal_args) + 1
            formal_args = formal_args[:-1] + (formal_args[-1].dtype,) * ndiff

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
        elif isinstance(formal, type) and issubclass(formal, types.Type):
            if isinstance(actual, type) and issubclass(actual, formal):
                # formal arg is a type class and actual arg is a subclass
                return True
            elif isinstance(actual, formal):
                # formal arg is a type class of which actual arg is an instance
                return True

    def append(self, value, sig):
        """
        Add a formal signature and its associated value.
        """
        assert isinstance(sig, tuple), (value, sig)
        self.versions.append((sig, value))
        self._cache.clear()

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

        # A mapping of installed registries to their loaders
        self._registries = {}
        # Declarations loaded from registries and other sources
        self._defns = defaultdict(OverloadSelector)
        self._getattrs = defaultdict(OverloadSelector)
        self._setattrs = defaultdict(OverloadSelector)
        self._casts = OverloadSelector()
        self._get_constants = OverloadSelector()
        # Other declarations
        self._generators = {}
        self.special_ops = {}
        self.cached_internal_func = {}
        self._pid = None

        self.data_model_manager = datamodel.default_manager

        # Initialize
        self.init()

    def init(self):
        """
        For subclasses to add initializer
        """

    def refresh(self):
        """
        Refresh context with new declarations from known registries.
        Useful for third-party extensions.
        """
        # Populate built-in registry
        from . import (arraymath, enumimpl, iterators, linalg, numbers,
                       optional, rangeobj, slicing, smartarray, tupleobj)
        try:
            from . import npdatetime
        except NotImplementedError:
            pass
        self.install_registry(builtin_registry)
        self.load_additional_registries()
        # Also refresh typing context, since @overload declarations can
        # affect it.
        self.typing_context.refresh()

    def load_additional_registries(self):
        """
        Load target-specific registries.  Can be overriden by subclasses.
        """

    def get_arg_packer(self, fe_args):
        return datamodel.ArgPacker(self.data_model_manager, fe_args)

    def get_data_packer(self, fe_types):
        return datamodel.DataPacker(self.data_model_manager, fe_types)

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
        try:
            loader = self._registries[registry]
        except KeyError:
            loader = RegistryLoader(registry)
            self._registries[registry] = loader
        self.insert_func_defn(loader.new_registrations('functions'))
        self._insert_getattr_defn(loader.new_registrations('getattrs'))
        self._insert_setattr_defn(loader.new_registrations('setattrs'))
        self._insert_cast_defn(loader.new_registrations('casts'))
        self._insert_get_constant_defn(loader.new_registrations('constants'))

    def insert_func_defn(self, defns):
        for impl, func, sig in defns:
            self._defns[func].append(impl, sig)

    def _insert_getattr_defn(self, defns):
        for impl, attr, sig in defns:
            self._getattrs[attr].append(impl, sig)

    def _insert_setattr_defn(self, defns):
        for impl, attr, sig in defns:
            self._setattrs[attr].append(impl, sig)

    def _insert_cast_defn(self, defns):
        for impl, sig in defns:
            self._casts.append(impl, sig)

    def _insert_get_constant_defn(self, defns):
        for impl, sig in defns:
            self._get_constants.append(impl, sig)

    def insert_user_function(self, func, fndesc, libs=()):
        impl = user_function(fndesc, libs)
        self._defns[func].append(impl, impl.signature)

    def add_user_function(self, func, fndesc, libs=()):
        if func not in self._defns:
            msg = "{func} is not a registered user function"
            raise KeyError(msg.format(func=func))
        impl = user_function(fndesc, libs)
        self._defns[func].append(impl, impl.signature)

    def insert_generator(self, genty, gendesc, libs=()):
        assert isinstance(genty, types.Generator)
        impl = user_generator(gendesc, libs)
        self._generators[genty] = gendesc, impl

    def remove_user_function(self, func):
        """
        Remove user function *func*.
        KeyError is raised if the function isn't known to us.
        """
        del self._defns[func]

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

    def get_constant_generic(self, builder, ty, val):
        """
        Return a LLVM constant representing value *val* of Numba type *ty*.
        """
        try:
            impl = self._get_constants.find((ty,))
            return impl(self, builder, ty, val)
        except NotImplementedError:
            raise NotImplementedError("cannot lower constant of type '%s'" % (ty,))

    def get_constant(self, ty, val):
        """
        Same as get_constant_generic(), but without specifying *builder*.
        Works only for simple types.
        """
        # HACK: pass builder=None to preserve get_constant() API
        return self.get_constant_generic(None, ty, val)

    def get_constant_undef(self, ty):
        lty = self.get_value_type(ty)
        return Constant.undef(lty)

    def get_constant_null(self, ty):
        lty = self.get_value_type(ty)
        return Constant.null(lty)

    def get_function(self, fn, sig):
        """
        Return the implementation of function *fn* for signature *sig*.
        The return value is a callable with the signature (builder, args).
        """
        sig = sig.as_function()
        if isinstance(fn, (types.Function, types.BoundFunction,
                           types.Dispatcher)):
            key = fn.get_impl_key(sig)
            overloads = self._defns[key]
        else:
            key = fn
            overloads = self._defns[key]

        try:
            return _wrap_impl(overloads.find(sig.args), self, sig)
        except NotImplementedError:
            pass
        if isinstance(fn, types.Type):
            # It's a type instance => try to find a definition for the type class
            try:
                return self.get_function(type(fn), sig)
            except NotImplementedError:
                # Raise exception for the type instance, for a better error message
                pass
        raise NotImplementedError("No definition for lowering %s%s" % (key, sig))

    def get_generator_desc(self, genty):
        """
        """
        return self._generators[genty][0]

    def get_generator_impl(self, genty):
        """
        """
        return self._generators[genty][1]

    def get_bound_function(self, builder, obj, ty):
        assert self.get_value_type(ty) == obj.type
        return obj

    def get_getattr(self, typ, attr):
        """
        Get the getattr() implementation for the given type and attribute name.
        The return value is a callable with the signature
        (context, builder, typ, val, attr).
        """
        if isinstance(typ, types.Module):
            # Implement getattr for module-level globals.
            # We are treating them as constants.
            # XXX We shouldn't have to retype this
            attrty = self.typing_context.resolve_module_constants(typ, attr)
            if attrty is None or isinstance(attrty, types.Dummy):
                # No implementation required for dummies (functions, modules...),
                # which are dealt with later
                return None
            else:
                pyval = getattr(typ.pymod, attr)
                llval = self.get_constant(attrty, pyval)
                def imp(context, builder, typ, val, attr):
                    return impl_ret_borrowed(context, builder, attrty, llval)
                return imp

        # Lookup specific getattr implementation for this type and attribute
        overloads = self._getattrs[attr]
        try:
            return overloads.find((typ,))
        except NotImplementedError:
            pass
        # Lookup generic getattr implementation for this type
        overloads = self._getattrs[None]
        try:
            return overloads.find((typ,))
        except NotImplementedError:
            pass

        raise NotImplementedError("No definition for lowering %s.%s" % (typ, attr))

    def get_setattr(self, attr, sig):
        """
        Get the setattr() implementation for the given attribute name
        and signature.
        The return value is a callable with the signature (builder, args).
        """
        assert len(sig.args) == 2
        typ = sig.args[0]
        valty = sig.args[1]

        def wrap_setattr(impl):
            def wrapped(builder, args):
                return impl(self, builder, sig, args, attr)
            return wrapped

        # Lookup specific setattr implementation for this type and attribute
        overloads = self._setattrs[attr]
        try:
            return wrap_setattr(overloads.find((typ, valty)))
        except NotImplementedError:
            pass
        # Lookup generic setattr implementation for this type
        overloads = self._setattrs[None]
        try:
            return wrap_setattr(overloads.find((typ, valty)))
        except NotImplementedError:
            pass

        raise NotImplementedError("No definition for lowering %s.%s = %s"
                                  % (typ, attr, valty))

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
        pair = self.make_helper(builder, ty, val)
        return pair.first

    def pair_second(self, builder, val, ty):
        """
        Extract the second element of a heterogenous pair.
        """
        pair = self.make_helper(builder, ty, val)
        return pair.second

    def cast(self, builder, val, fromty, toty):
        """
        Cast a value of type *fromty* to type *toty*.
        This implements implicit conversions as can happen due to the
        granularity of the Numba type system, or lax Python semantics.
        """
        if fromty == toty or toty == types.Any:
            return val
        try:
            impl = self._casts.find((fromty, toty))
            return impl(self, builder, fromty, toty, val)
        except NotImplementedError:
            raise NotImplementedError(
                "Cannot cast %s to %s: %s" % (fromty, toty, val))

    def generic_compare(self, builder, key, argtypes, args):
        """
        Compare the given LLVM values of the given Numba types using
        the comparison *key* (e.g. '==').  The values are first cast to
        a common safe conversion type.
        """
        at, bt = argtypes
        av, bv = args
        ty = self.typing_context.unify_types(at, bt)
        assert ty is not None
        cav = self.cast(builder, av, at, ty)
        cbv = self.cast(builder, bv, bt, ty)
        cmpsig = typing.signature(types.boolean, ty, ty)
        cmpfunc = self.get_function(key, cmpsig)
        return cmpfunc(builder, (cav, cbv))

    def make_optional_none(self, builder, valtype):
        optval = self.make_helper(builder, types.Optional(valtype))
        optval.valid = cgutils.false_bit
        return optval._getvalue()

    def make_optional_value(self, builder, valtype, value):
        optval = self.make_helper(builder, types.Optional(valtype))
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
        mod = builder.module
        cstring = GENERIC_POINTER
        fnty = Type.function(Type.int(), [cstring])
        puts = mod.get_or_insert_function(fnty, "puts")
        return builder.call(puts, [text])

    def debug_print(self, builder, text):
        mod = builder.module
        cstr = self.insert_const_string(mod, str(text))
        self.print_string(builder, cstr)

    def printf(self, builder, format_string, *args):
        mod = builder.module
        if isinstance(format_string, str):
            cstr = self.insert_const_string(mod, format_string)
        else:
            cstr = format_string
        fnty = Type.function(Type.int(), (GENERIC_POINTER,), var_arg=True)
        fn = mod.get_or_insert_function(fnty, "printf")
        return builder.call(fn, (cstr,) + tuple(args))

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

    def compile_subroutine_no_cache(self, builder, impl, sig, locals={}, flags=None):
        """
        Invoke the compiler to compile a function to be used inside a
        nopython function, but without generating code to call that
        function.

        Note this context's flags are not inherited.
        """
        # Compile
        from numba import compiler

        codegen = self.codegen()
        library = codegen.create_library(impl.__name__)
        if flags is None:
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
        cache_key = (impl.__code__, sig, type(self.error_model))
        if impl.__closure__:
            # XXX This obviously won't work if a cell's value is
            # unhashable.
            cache_key += tuple(c.cell_contents for c in impl.__closure__)
        ty = self.cached_internal_func.get(cache_key)
        if ty is None:
            cres = self.compile_subroutine_no_cache(builder, impl, sig,
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
        """
        Given the function descriptor of an internally compiled function,
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

    def get_helper_class(self, typ, kind='value'):
        """
        Get a helper class for the given *typ*.
        """
        # XXX handle all types: complex, array, etc.
        # XXX should it be a method on the model instead? this would allow a default kind...
        return cgutils.create_struct_proxy(typ, kind)

    def _make_helper(self, builder, typ, value=None, ref=None, kind='value'):
        cls = self.get_helper_class(typ, kind)
        return cls(self, builder, value=value, ref=ref)

    def make_helper(self, builder, typ, value=None, ref=None):
        """
        Get a helper object to access the *typ*'s members,
        for the given value or reference.
        """
        return self._make_helper(builder, typ, value, ref, kind='value')

    def make_data_helper(self, builder, typ, ref=None):
        """
        As make_helper(), but considers the value as stored in memory,
        rather than a live value.
        """
        return self._make_helper(builder, typ, ref=ref, kind='data')

    def make_array(self, typ):
        return arrayobj.make_array(typ)

    def populate_array(self, arr, **kwargs):
        """
        Populate array structure.
        """
        return arrayobj.populate_array(arr, **kwargs)

    def make_complex(self, builder, typ, value=None):
        """
        Get a helper object to access the given complex numbers' members.
        """
        assert isinstance(typ, types.Complex), typ
        return self.make_helper(builder, typ, value)

    def make_tuple(self, builder, typ, values):
        """
        Create a tuple of the given *typ* containing the *values*.
        """
        tup = self.get_constant_undef(typ)
        for i, val in enumerate(values):
            tup = builder.insert_value(tup, val, i)
        return tup

    def make_constant_array(self, builder, typ, ary):
        """
        Create an array structure reifying the given constant array.
        A low-level contiguous array constant is created in the LLVM IR.
        """
        assert typ.layout == 'C'                # assumed in typeinfer.py

        # Handle data: reify the flattened array in "C" order as a
        # global array of bytes.
        flat = ary.flatten()
        # Note: we use `bytearray(flat.data)` instead of `bytearray(flat)` to
        #       workaround issue #1850 which is due to numpy issue #3147
        consts = Constant.array(Type.int(8), bytearray(flat.data))
        data = cgutils.global_constant(builder, ".const.array.data", consts)

        # Handle shape
        llintp = self.get_value_type(types.intp)
        shapevals = [self.get_constant(types.intp, s) for s in ary.shape]
        cshape = Constant.array(llintp, shapevals)

        # Handle strides: use strides of the equivalent C-contiguous array.
        contig = np.ascontiguousarray(ary)
        stridevals = [self.get_constant(types.intp, s) for s in contig.strides]
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
        return lc.Module(name)

    def _require_nrt(self):
        if not self.enable_nrt:
            raise RuntimeError("Require NRT")

    def nrt_meminfo_alloc(self, builder, size):
        """
        Allocate a new MemInfo with a data payload of `size` bytes.

        A pointer to the MemInfo is returned.
        """
        self._require_nrt()

        mod = builder.module
        fnty = llvmir.FunctionType(void_ptr,
                                   [self.get_value_type(types.intp)])
        fn = mod.get_or_insert_function(fnty, name="NRT_MemInfo_alloc_safe")
        fn.return_value.add_attribute("noalias")
        return builder.call(fn, [size])

    def nrt_meminfo_alloc_dtor(self, builder, size, dtor):
        self._require_nrt()

        mod = builder.module
        ll_void_ptr = self.get_value_type(types.voidptr)
        fnty = llvmir.FunctionType(llvmir.IntType(8).as_pointer(),
                                   [self.get_value_type(types.intp),
                                    ll_void_ptr])
        fn = mod.get_or_insert_function(fnty,
                                        name="NRT_MemInfo_alloc_dtor_safe")
        fn.return_value.add_attribute("noalias")
        return builder.call(fn, [size, builder.bitcast(dtor, ll_void_ptr)])

    def nrt_meminfo_alloc_aligned(self, builder, size, align):
        """
        Allocate a new MemInfo with an aligned data payload of `size` bytes.
        The data pointer is aligned to `align` bytes.  `align` can be either
        a Python int or a LLVM uint32 value.

        A pointer to the MemInfo is returned.
        """
        self._require_nrt()

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

    def nrt_meminfo_new_varsize(self, builder, size):
        """
        Allocate a MemInfo pointing to a variable-sized data area.  The area
        is separately allocated (i.e. two allocations are made) so that
        re-allocating it doesn't change the MemInfo's address.

        A pointer to the MemInfo is returned.
        """
        self._require_nrt()

        mod = builder.module
        fnty = llvmir.FunctionType(void_ptr,
                                   [self.get_value_type(types.intp)])
        fn = mod.get_or_insert_function(fnty, name="NRT_MemInfo_new_varsize")
        fn.return_value.add_attribute("noalias")
        return builder.call(fn, [size])

    def nrt_meminfo_varsize_alloc(self, builder, meminfo, size):
        """
        Allocate a new data area for a MemInfo created by nrt_meminfo_new_varsize().
        The new data pointer is returned, for convenience.

        Contrary to realloc(), this always allocates a new area and doesn't
        copy the old data.  This is useful if resizing a container needs
        more than simply copying the data area (e.g. for hash tables).

        The old pointer will have to be freed with nrt_meminfo_varsize_free().
        """
        return self._call_nrt_varsize_alloc(builder, meminfo, size,
                                            "NRT_MemInfo_varsize_alloc")

    def nrt_meminfo_varsize_realloc(self, builder, meminfo, size):
        """
        Reallocate a data area allocated by nrt_meminfo_new_varsize().
        The new data pointer is returned, for convenience.
        """
        return self._call_nrt_varsize_alloc(builder, meminfo, size,
                                            "NRT_MemInfo_varsize_realloc")

    def nrt_meminfo_varsize_free(self, builder, meminfo, ptr):
        """
        Free a memory area allocated for a NRT varsize object.
        Note this does *not* free the NRT object itself!
        """
        self._require_nrt()

        mod = builder.module
        fnty = llvmir.FunctionType(llvmir.VoidType(),
                                   [void_ptr, void_ptr])
        fn = mod.get_or_insert_function(fnty, name="NRT_MemInfo_varsize_free")
        return builder.call(fn, (meminfo, ptr))

    def _call_nrt_varsize_alloc(self, builder, meminfo, size, funcname):
        self._require_nrt()

        mod = builder.module
        fnty = llvmir.FunctionType(void_ptr,
                                   [void_ptr, self.get_value_type(types.intp)])
        fn = mod.get_or_insert_function(fnty, name=funcname)
        fn.return_value.add_attribute("noalias")
        return builder.call(fn, [meminfo, size])

    def nrt_meminfo_data(self, builder, meminfo):
        """
        Given a MemInfo pointer, return a pointer to the allocated data
        managed by it.  This works for MemInfos allocated with all the
        above methods.
        """
        self._require_nrt()

        from numba.runtime.atomicops import meminfo_data_ty

        mod = builder.module
        fn = mod.get_or_insert_function(meminfo_data_ty,
                                        name="NRT_MemInfo_data_fast")
        return builder.call(fn, [meminfo])

    def _call_nrt_incref_decref(self, builder, root_type, typ, value,
                                funcname, getters=()):
        self._require_nrt()

        from numba.runtime.atomicops import incref_decref_ty

        data_model = self.data_model_manager[typ]

        members = data_model.traverse(builder)
        for mtyp, getter in members:
            self._call_nrt_incref_decref(builder, root_type, mtyp, value,
                                         funcname, getters + (getter,))

        if data_model.has_nrt_meminfo():
            # Call the chain of getters to compute the member value
            for getter in getters:
                value = getter(value)
            try:
                meminfo = data_model.get_nrt_meminfo(builder, value)
            except NotImplementedError as e:
                raise NotImplementedError("%s: %s" % (root_type, str(e)))
            assert meminfo is not None  # since has_nrt_meminfo()
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
    """
    A wrapper object to call an implementation function with some predefined
    (context, signature) arguments.
    The wrapper also forwards attribute queries, which is important.
    """

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
