from __future__ import print_function, absolute_import
from collections import defaultdict
import functools
import types as pytypes
import numpy
from numba import types
from numba.typeconv import rules
from . import templates
# Initialize declarations
from . import builtins, mathdecl, npydecl, operatordecl
from numba import numpy_support, utils
from . import ctypes_utils, cffi_utils

class BaseContext(object):
    """A typing context for storing function typing constrain template.
    """

    def __init__(self):
        self.functions = defaultdict(list)
        self.attributes = {}
        self.globals = utils.UniqueDict()
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
        if isinstance(func, types.Function):
            return func.template(self).apply(args, kws)

        if isinstance(func, types.Dispatcher):
            if kws:
                raise TypeError("kwargs not supported")
            if not func.overloaded.is_compiling:
                # Avoid compiler re-entrant
                fnobj = func.overloaded.compile(tuple(args))
            else:
                try:
                    fnobj = func.overloaded.get_overload(tuple(args))
                except KeyError:
                    return None
            ty = self.globals[fnobj]
            return self.resolve_function_type(ty, args, kws)

        defns = self.functions[func]
        for defn in defns:
            res = defn.apply(args, kws)
            if res is not None:
                return res

    def resolve_getattr(self, value, attr):
        if isinstance(value, types.Record):
            return value.typeof(attr)

        try:
            attrinfo = self.attributes[value]
        except KeyError:
            if value.is_parametric:
                attrinfo = self.attributes[type(value)]
            elif isinstance(value, types.Module):
                attrty = self.resolve_module_constants(value, attr)
                if attrty is not None:
                    return attrty
            else:
                raise

        return attrinfo.resolve(value, attr)

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
            if ty in types.number_domain:
                return ty

    def resolve_value_type(self, val):
        """
        Return the numba type of a Python value
        Return None if fail to type.
        """
        if val is True or val is False:
            return types.boolean

        elif isinstance(val, utils.INT_TYPES + (float,)):
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

        elif numpy_support.is_arrayscalar(val):
            return numpy_support.map_arrayscalar_type(val)

        elif numpy_support.is_array(val):
            ary = val
            dtype = numpy_support.from_dtype(ary.dtype)
            # Force C contiguous
            return types.Array(dtype, ary.ndim, 'C')

        elif ctypes_utils.is_ctypes_funcptr(val):
            cfnptr = val
            return ctypes_utils.make_function_type(cfnptr)

        elif cffi_utils.SUPPORTED and cffi_utils.is_cffi_func(val):
            return cffi_utils.make_function_type(val)

        elif (cffi_utils.SUPPORTED and
                  isinstance(val, cffi_utils.ExternCFunction)):
            return val

        elif type(val) is type and issubclass(val, BaseException):
            return types.exception_type

        else:
            try:
                # Try to look up target specific typing information
                return self.get_global_type(val)
            except KeyError:
                pass

        return None

    def get_global_type(self, gv):
        try:
            return self.globals[gv]
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

    def insert_global(self, gv, gty):
        self.globals[gv] = gty

    def insert_attributes(self, at):
        key = at.key
        assert key not in self.attributes, "Duplicated attributes template"
        self.attributes[key] = at

    def insert_function(self, ft):
        key = ft.key
        self.functions[key].append(ft)

    def insert_overloaded(self, overloaded):
        self.globals[overloaded] = types.Dispatcher(overloaded)

    def insert_user_function(self, fn, ft):
        """Insert a user function.

        Args
        ----
        - fn:
            object used as callee
        - ft:
            function template
        """
        self.globals[fn] = types.Function(ft)

    def extend_user_function(self, fn, ft):
        """ Insert of extend a user function.

        Args
        ----
        - fn:
            object used as callee
        - ft:
            function template
        """
        if fn in self.globals:
            self.globals[fn].extend(ft)
        else:
            self.insert_user_function(fn, ft)

    def insert_class(self, cls, attrs):
        clsty = types.Object(cls)
        at = templates.ClassAttrTemplate(self, clsty, attrs)
        self.insert_attributes(at)

    def type_compatibility(self, fromty, toty):
        """
        Returns None or a string describing the conversion e.g. exact, promote,
        unsafe, safe, tuple-coerce
        """
        if fromty == toty:
            return 'exact'

        elif (isinstance(fromty, types.UniTuple) and
                  isinstance(toty, types.UniTuple) and
                      len(fromty) == len(toty)):
            return self.type_compatibility(fromty.dtype, toty.dtype)

        elif (types.is_int_tuple(fromty) and types.is_int_tuple(toty) and
                      len(fromty) == len(toty)):
            return 'int-tuple-coerce'

        return self.tm.check_compatible(fromty, toty)

    def unify_types(self, *typelist):
        # Sort the type list according to bit width before doing
        # pairwise unification (with thanks to aterrel).
        def keyfunc(obj): return getattr(obj, 'bitwidth', hash(obj))
        return functools.reduce(
            self.unify_pairs, sorted(typelist, key=keyfunc))

    def unify_pairs(self, first, second):
        """
        Choose PyObject type as the abstract if we fail to determine a concrete
        type.
        """
        # TODO: should add an option to reject unsafe type conversion
        d = self.type_compatibility(fromty=first, toty=second)
        if d is None:
            # Complex is not allowed to downcast implicitly.
            # Need to try the other direction of implicit cast to find the
            # most general type of the two.
            first, second = second, first   # swap operand order
            d = self.type_compatibility(fromty=first, toty=second)

        if d is None:
            return types.pyobject
        elif d == 'exact':
            # Same type
            return first
        elif d == 'promote':
            return second
        elif d in ('safe', 'unsafe'):
            assert first in types.number_domain
            assert second in types.number_domain
            a = numpy.dtype(str(first))
            b = numpy.dtype(str(second))
            # Just use NumPy coercion rules
            sel = numpy.promote_types(a, b)
            # Convert NumPy dtype back to Numba types
            return getattr(types, str(sel))
        elif d in 'int-tuple-coerce':
            return types.UniTuple(dtype=types.intp, count=len(first))
        else:
            raise Exception("type_compatibility returned %s" % d)


class Context(BaseContext):
    def init(self):
        self.install(mathdecl.registry)
        self.install(npydecl.registry)
        self.install(operatordecl.registry)


def new_method(fn, sig):
    name = "UserFunction_%s" % fn
    ft = templates.make_concrete_template(name, fn, [sig])
    return types.Method(ft, this=sig.recvr)


