from __future__ import print_function, absolute_import

from collections import defaultdict
import functools
import types as pytypes
import sys
import weakref

import numpy

from numba import types
from numba.typeconv import rules
from . import templates

# Initialize declarations
from . import (
    builtins, cmathdecl, mathdecl, npdatetime, npydecl, operatordecl,
    randomdecl)
from numba import numpy_support, utils
from . import ctypes_utils, cffi_utils, bufproto


class BaseContext(object):
    """A typing context for storing function typing constrain template.
    """

    def __init__(self):
        self.functions = defaultdict(list)
        self.attributes = {}
        self._globals = utils.UniqueDict()
        self.tm = rules.default_type_manager
        self._load_builtins()
        self.init()

    def init(self):
        pass

    def get_number_type(self, num):
        if isinstance(num, utils.INT_TYPES):
            nbits = utils.bit_length(num)
            if nbits < 32:
                typ = types.int32
            elif nbits < 64:
                typ = types.int64
            elif nbits == 64 and num >= 0:
                typ = types.uint64
            else:
                raise ValueError("Int value is too large: %s" % num)
            return typ
        elif isinstance(num, float):
            return types.float64
        else:
            raise NotImplementedError(type(num), num)

    def resolve_function_type(self, func, args, kws):
        """
        Resolve function type *func* for argument types *args* and *kws*.
        A signature is returned.
        """
        if isinstance(func, types.Callable):
            return func.get_call_type(self, args, kws)

        defns = self.functions[func]
        for defn in defns:
            res = defn.apply(args, kws)
            if res is not None:
                return res

    def resolve_getattr(self, value, attr):
        if isinstance(value, types.Record):
            ret = value.typeof(attr)
            assert ret
            return ret

        try:
            attrinfo = self.attributes[value]
        except KeyError:
            if value.is_parametric:
                attrinfo = self.attributes[type(value)]
            elif isinstance(value, types.Module):
                attrty = self.resolve_module_constants(value, attr)
                if attrty is not None:
                    return attrty
                raise
            else:
                raise

        ret = attrinfo.resolve(value, attr)
        if ret is None:
            raise KeyError(attr)
        return ret

    def resolve_setitem(self, target, index, value):
        args = target, index, value
        kws = ()
        return self.resolve_function_type("setitem", args, kws)

    def resolve_setattr(self, target, attr, value):
        if isinstance(target, types.Record):
            expectedty = target.typeof(attr)
            if self.type_compatibility(value, expectedty) is not None:
                return templates.signature(types.void, target, value)

    def resolve_module_constants(self, typ, attr):
        """Resolve module-level global constants
        Return None or the attribute type
        """
        if isinstance(typ, types.Module):
            attrval = getattr(typ.pymod, attr)
            ty = self.resolve_value_type(attrval)
            return ty

    def resolve_argument_type(self, val):
        """
        Return the numba type of a Python value that is being used
        as a function argument.  Integer types will all be considered
        int64, regardless of size.  Numpy arrays will be accepted in
        "C" or "F" layout.

        Unknown types will be mapped to pyobject.
        """
        # Force all integers (except bools) to be 64-bit
        if isinstance(val, utils.INT_TYPES) and not isinstance(val, bool):
            return types.int64

        tp = self.resolve_data_type(val)
        if tp is None:
            tp = getattr(val, "_numba_type_", types.pyobject)
        return tp

    def resolve_data_type(self, val):
        """
        Return the numba type of a Python value representing data
        (e.g. a number or an array, but not more sophisticated types
         such as functions, etc.)

        This function can return None to if it cannot decide.
        """
        if val is True or val is False:
            return types.boolean

        # Under 2.x, we must guard against numpy scalars (np.intXY
        # subclasses Python int but get_number_type() wouldn't infer the
        # right bit width -- perhaps it should?).
        elif (not isinstance(val, numpy.number)
              and isinstance(val, utils.INT_TYPES + (float,))):
            return self.get_number_type(val)

        elif val is None:
            return types.none

        elif isinstance(val, str):
            return types.string

        elif isinstance(val, complex):
            return types.complex128

        elif isinstance(val, tuple):
            tys = [self.resolve_value_type(v) for v in val]
            distinct_types = set(tys)
            if len(distinct_types) == 1:
                return types.UniTuple(tys[0], len(tys))
            else:
                return types.Tuple(tys)

        elif ctypes_utils.is_ctypes_funcptr(val):
            return ctypes_utils.make_function_type(val)

        elif cffi_utils.SUPPORTED and cffi_utils.is_cffi_func(val):
            return cffi_utils.make_function_type(val)

        elif numpy_support.is_array(val):
            ary = val
            try:
                dtype = numpy_support.from_dtype(ary.dtype)
            except NotImplementedError:
                return
            layout = numpy_support.map_layout(ary)
            return types.Array(dtype, ary.ndim, layout)

        elif sys.version_info >= (2, 7) and not isinstance(val, numpy.generic):
            try:
                m = memoryview(val)
            except TypeError:
                pass
            else:
                # Object has the buffer protocol
                try:
                    dtype = bufproto.decode_pep3118_format(m.format, m.itemsize)
                except ValueError:
                    pass
                else:
                    type_class = bufproto.get_type_class(type(val))
                    layout = bufproto.infer_layout(m)
                    return type_class(dtype, m.ndim, layout=layout,
                                      readonly=m.readonly)

        else:
            # Matching here is quite broad, so we have to do it after
            # the more specific matches above.
            try:
                return numpy_support.map_arrayscalar_type(val)
            except NotImplementedError:
                pass

        return

    def resolve_value_type(self, val):
        """
        Return the numba type of a Python value
        Return None if fail to type.
        """
        tp = self.resolve_data_type(val)
        if tp is not None:
            return tp

        if isinstance(val, types.ExternalFunction):
            return val

        if isinstance(val, type) and issubclass(val, BaseException):
            return types.ExceptionType(val)

        if isinstance(val, numpy.dtype):
            tp = numpy_support.from_dtype(val)
            return types.DType(tp)

        try:
            # Try to look up target specific typing information
            return self.get_global_type(val)
        except KeyError:
            pass

        return None

    def get_global_type(self, gv):
        try:
            return self._lookup_global(gv)
        except KeyError:
            if isinstance(gv, pytypes.ModuleType):
                return types.Module(gv)
            else:
                raise

    def _load_builtins(self):
        self.install(templates.builtin_registry)

    def install(self, registry):
        for ftcls in registry.functions:
            self.insert_function(ftcls(self))
        for ftcls in registry.attributes:
            self.insert_attributes(ftcls(self))
        for gv, gty in registry.globals:
            self.insert_global(gv, gty)

    def _lookup_global(self, gv):
        """
        Look up the registered type for global value *gv*.
        """
        try:
            gv = weakref.ref(gv)
        except TypeError:
            pass
        return self._globals[gv]

    def _insert_global(self, gv, gty):
        """
        Register type *gty* for value *gv*.  Only a weak reference
        to *gv* is kept, if possible.
        """
        def on_disposal(wr):
            self._globals.pop(wr)
        try:
            gv = weakref.ref(gv, on_disposal)
        except TypeError:
            pass
        self._globals[gv] = gty

    def insert_global(self, gv, gty):
        self._insert_global(gv, gty)

    def insert_attributes(self, at):
        key = at.key
        assert key not in self.attributes, "Duplicated attributes template"
        self.attributes[key] = at

    def insert_function(self, ft):
        key = ft.key
        self.functions[key].append(ft)

    def insert_overloaded(self, overloaded):
        self._insert_global(overloaded, types.Dispatcher(overloaded))

    def insert_user_function(self, fn, ft):
        """Insert a user function.

        Args
        ----
        - fn:
            object used as callee
        - ft:
            function template
        """
        self._insert_global(fn, types.Function(ft))

    def insert_class(self, cls, attrs):
        clsty = types.Object(cls)
        at = templates.ClassAttrTemplate(self, clsty, attrs)
        self.insert_attributes(at)

    def type_compatibility(self, fromty, toty):
        """
        Returns None or a string describing the conversion e.g. exact, promote,
        unsafe, safe.
        """
        if fromty == toty:
            return 'exact'

        elif (isinstance(fromty, types.UniTuple) and
                  isinstance(toty, types.UniTuple) and
                      len(fromty) == len(toty)):
            return self.type_compatibility(fromty.dtype, toty.dtype)

        else:
            return self.tm.check_compatible(fromty, toty)

    def unify_types(self, *typelist):
        # Sort the type list according to bit width before doing
        # pairwise unification (with thanks to aterrel).
        def keyfunc(obj):
            """Uses bitwidth to order numeric-types.
            Fallback to hash() for arbitary ordering.
            """
            return getattr(obj, 'bitwidth', hash(obj))

        return functools.reduce(
            self.unify_pairs, sorted(typelist, key=keyfunc))

    def unify_pairs(self, first, second):
        """
        Choose PyObject type as the abstract if we fail to determine a concrete
        type.
        """
        if first == second:
            return first

        # Types with special coercion rule
        first_coerce = first.coerce(self, second)
        second_coerce = second.coerce(self, first)

        if first_coerce is not NotImplemented:
            return first_coerce

        elif second_coerce is not NotImplemented:
            return second_coerce

        # TODO: should add an option to reject unsafe type conversion

        # Types with simple coercion rule
        forward = self.type_compatibility(fromty=first, toty=second)
        backward = self.type_compatibility(fromty=second, toty=first)

        strong = ('exact', 'promote')
        weak = ('safe', 'unsafe')
        if forward in strong:
            return second
        elif backward in strong:
            return first
        elif forward is None and backward is None:
            return types.pyobject
        elif forward in weak or backward in weak:
            # Use numpy to pick a type that
            if first in types.number_domain and second in types.number_domain:
                a = numpy.dtype(str(first))
                b = numpy.dtype(str(second))
                sel = numpy.promote_types(a, b)
                return getattr(types, str(sel))


        # Failed to unify
        msg = ("Cannot unify {{{first}, {second}}}\n"
               "{first}->{second}::{forward}\n"
               "{second}->{first}::{backward} ")
        raise AssertionError(msg.format(**locals()))


class Context(BaseContext):
    def init(self):
        self.install(cmathdecl.registry)
        self.install(mathdecl.registry)
        self.install(npydecl.registry)
        self.install(operatordecl.registry)
        self.install(randomdecl.registry)


def new_method(fn, sig):
    name = "UserFunction_%s" % fn
    ft = templates.make_concrete_template(name, fn, [sig])
    return types.Method(ft, this=sig.recvr)

