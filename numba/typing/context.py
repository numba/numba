from __future__ import print_function, absolute_import

from collections import defaultdict
import types as pytypes
import weakref

from numba import types
from numba.typeconv import Conversion, rules
from . import templates
from .typeof import typeof, Purpose

# Initialize declarations
from . import (
    builtins, arraydecl, cmathdecl, listdecl, mathdecl, npdatetime, npydecl,
    operatordecl, randomdecl)
from numba import utils
from . import ctypes_utils, cffi_utils, bufproto


class Rating(object):
    __slots__ = 'promote', 'safe_convert', "unsafe_convert"

    def __init__(self):
        self.promote = 0
        self.safe_convert = 0
        self.unsafe_convert = 0

    def astuple(self):
        """Returns a tuple suitable for comparing with the worse situation
        start first.
        """
        return (self.unsafe_convert, self.safe_convert, self.promote)

    def __add__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        rsum = Rating()
        rsum.promote = self.promote + other.promote
        rsum.safe_convert = self.safe_convert + other.safe_convert
        rsum.unsafe_convert = self.unsafe_convert + other.unsafe_convert
        return rsum


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

    def explain_function_type(self, func):
        """
        Returns a string description of the type of a function
        """
        desc = []
        defns = []
        param = False
        if isinstance(func, types.Callable):
            sigs, param = func.get_call_signatures()
            defns.extend(sigs)

        elif func in self.functions:
            for tpl in self.functions[func]:
                param = param or hasattr(tpl, 'generic')
                defns.extend(getattr(tpl, 'cases', []))

        else:
            msg = "No type info available for {func} as a callable."
            desc.append(msg.format(func=func))
            return desc

        if defns:
            desc = ['Known signatures:']
            for sig in defns:
                desc.append(' * {0}'.format(sig))

        if param:
            desc.append(' * parameterized')

        return '\n'.join(desc)

    def resolve_function_type(self, func, args, kws):
        """
        Resolve function type *func* for argument types *args* and *kws*.
        A signature is returned.
        """
        defns = self.functions[func]
        for defn in defns:
            res = defn.apply(args, kws)
            if res is not None:
                return res

        if isinstance(func, types.Type):
            # If it's a type, it may support a __call__ method
            try:
                func_type = self.resolve_getattr(func, "__call__")
            except KeyError:
                pass
            else:
                # The function has a __call__ method, type its call.
                return self.resolve_function_type(func_type, args, kws)

        if isinstance(func, types.Callable):
            # XXX fold this into the __call__ attribute logic?
            return func.get_call_type(self, args, kws)

    def resolve_getattr(self, value, attr):
        try:
            attrinfo = self.attributes[value]
        except KeyError:
            for cls in type(value).__mro__:
                if cls in self.attributes:
                    attrinfo = self.attributes[cls]
                    break
            else:
                if isinstance(value, types.Module):
                    attrty = self.resolve_module_constants(value, attr)
                    if attrty is not None:
                        return attrty
                raise

        ret = attrinfo.resolve(value, attr)
        if ret is None:
            raise KeyError(attr)
        return ret

    def resolve_setattr(self, target, attr, value):
        if isinstance(target, types.Record):
            expectedty = target.typeof(attr)
            if self.can_convert(value, expectedty) is not None:
                return templates.signature(types.void, target, value)

    def resolve_static_getitem(self, value, index):
        assert not isinstance(index, types.Type), index
        args = value, index
        kws = ()
        return self.resolve_function_type("static_getitem", args, kws)

    def resolve_static_setitem(self, target, index, value):
        assert not isinstance(index, types.Type), index
        args = target, index, value
        kws = ()
        return self.resolve_function_type("static_setitem", args, kws)

    def resolve_setitem(self, target, index, value):
        assert isinstance(index, types.Type), index
        args = target, index, value
        kws = ()
        return self.resolve_function_type("setitem", args, kws)

    def resolve_delitem(self, target, index):
        args = target, index
        kws = ()
        return self.resolve_function_type("delitem", args, kws)

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
        int64, regardless of size.

        None is returned for unsupported types.
        """
        return typeof(val, Purpose.argument)

    def resolve_value_type(self, val):
        """
        Return the numba type of a Python value that is being used
        as a runtime constant.
        None is returned for unsupported types.
        """
        tp = typeof(val, Purpose.constant)
        if tp is not None:
            return tp

        if isinstance(val, (types.ExternalFunction, types.NumbaFunction)):
            return val

        if isinstance(val, type):
            if issubclass(val, BaseException):
                return types.ExceptionClass(val)
            if issubclass(val, tuple) and hasattr(val, "_asdict"):
                return types.NamedTupleClass(val)

        try:
            # Try to look up target specific typing information
            return self._get_global_type(val)
        except KeyError:
            pass

        return None

    def _get_global_type(self, gv):
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
        def on_disposal(wr, pop=self._globals.pop):
            # pop() is pre-looked up to avoid a crash late at shutdown on 3.5
            # (https://bugs.python.org/issue25217)
            pop(wr)
        try:
            gv = weakref.ref(gv, on_disposal)
        except TypeError:
            pass
        self._globals[gv] = gty

    def insert_global(self, gv, gty):
        self._insert_global(gv, gty)

    def insert_attributes(self, at):
        key = at.key
        assert key not in self.attributes, "Duplicated attributes template %r" % (key,)
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

    def can_convert(self, fromty, toty):
        """
        Check whether conversion is possible from *fromty* to *toty*.
        If successful, return a numba.typeconv.Conversion instance;
        otherwise None is returned.
        """
        if fromty == toty:
            return Conversion.exact
        else:
            # First check with the type manager (some rules are registered
            # at startup there, see numba.typeconv.rules)
            conv = self.tm.check_compatible(fromty, toty)
            if conv is not None:
                return conv

            # Fall back on type-specific rules
            forward = fromty.can_convert_to(self, toty)
            backward = toty.can_convert_from(self, fromty)
            if backward is None:
                return forward
            elif forward is None:
                return backward
            else:
                return min(forward, backward)

    def _rate_arguments(self, actualargs, formalargs):
        """
        Rate the actual arguments for compatibility against the formal
        arguments.  A Rating instance is returned, or None if incompatible.
        """
        if len(actualargs) != len(formalargs):
            return None
        rate = Rating()
        for actual, formal in zip(actualargs, formalargs):
            conv = self.can_convert(actual, formal)
            if conv is None:
                return None

            if conv == Conversion.promote:
                rate.promote += 1
            elif conv == Conversion.safe:
                rate.safe_convert += 1
            elif conv == Conversion.unsafe:
                rate.unsafe_convert += 1
            elif conv == Conversion.exact:
                pass
            else:
                raise Exception("unreachable", conv)

        return rate

    def install_possible_conversions(self, actualargs, formalargs):
        """
        Install possible conversions from the actual argument types to
        the formal argument types in the C++ type manager.
        Return True if all arguments can be converted.
        """
        if len(actualargs) != len(formalargs):
            return False
        for actual, formal in zip(actualargs, formalargs):
            if self.tm.check_compatible(actual, formal) is not None:
                # This conversion is already known
                continue
            conv = self.can_convert(actual, formal)
            if conv is None:
                return False
            assert conv is not Conversion.exact
            self.tm.set_compatible(actual, formal, conv)
        return True

    def resolve_overload(self, key, cases, args, kws,
                         allow_ambiguous=True):
        """
        Given actual *args* and *kws*, find the best matching
        signature in *cases*, or None if none matches.
        *key* is used for error reporting purposes.
        If *allow_ambiguous* is False, a tie in the best matches
        will raise an error.
        """
        assert not kws, "Keyword arguments are not supported, yet"
        # Rate each case
        candidates = []
        for case in cases:
            if len(args) == len(case.args):
                rating = self._rate_arguments(args, case.args)
                if rating is not None:
                    candidates.append((rating.astuple(), case))

        # Find the best case
        candidates.sort(key=lambda i: i[0])
        if candidates:
            best_rate, best = candidates[0]
            if not allow_ambiguous:
                # Find whether there is a tie and if so, raise an error
                tied = []
                for rate, case in candidates:
                    if rate != best_rate:
                        break
                    tied.append(case)
                if len(tied) > 1:
                    args = (key, args, '\n'.join(map(str, tied)))
                    msg = "Ambiguous overloading for %s %s:\n%s" % args
                    raise TypeError(msg)
            # Simply return the best matching candidate in order.
            # If there is a tie, since list.sort() is stable, the first case
            # in the original order is returned.
            # (this can happen if e.g. a function template exposes
            #  (int32, int32) -> int32 and (int64, int64) -> int64,
            #  and you call it with (int16, int16) arguments)
            return best

    def unify_types(self, *typelist):
        # Sort the type list according to bit width before doing
        # pairwise unification (with thanks to aterrel).
        def keyfunc(obj):
            """Uses bitwidth to order numeric-types.
            Fallback to stable, deterministic sort.
            """
            return getattr(obj, 'bitwidth', 0)

        typelist = sorted(typelist, key=keyfunc)
        unified = typelist[0]
        for tp in typelist[1:]:
            unified = self.unify_pairs(unified, tp)
            if unified is None:
                break
        return unified

    def unify_pairs(self, first, second):
        """
        Try to unify the two given types.  A third type is returned,
        or pyobject in case of failure.
        """
        if first == second:
            return first
        
        if first is types.undefined:
            return second
        elif second is types.undefined:
            return first

        # Types with special unification rules
        unified = first.unify(self, second)
        if unified is not None:
            return unified

        unified = second.unify(self, first)
        if unified is not None:
            return unified

        # Other types with simple conversion rules
        conv = self.can_convert(fromty=first, toty=second)
        if conv is not None and conv <= Conversion.safe:
            # Can convert from first to second
            return second

        conv = self.can_convert(fromty=second, toty=first)
        if conv is not None and conv <= Conversion.safe:
            # Can convert from second to first
            return first

        # Cannot unify
        return types.pyobject


class Context(BaseContext):
    def init(self):
        self.install(cmathdecl.registry)
        self.install(listdecl.registry)
        self.install(mathdecl.registry)
        self.install(npydecl.registry)
        self.install(operatordecl.registry)
        self.install(randomdecl.registry)
        self.install(cffi_utils.registry)

