from __future__ import print_function
from collections import defaultdict
import functools
import numpy
from numba import types, utils
from numba.typeconv import rules
from . import templates
# Initialize declarations
from . import builtins, mathdecl, npydecl, operatordecl


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

    def get_global_type(self, gv):
        return self.globals[gv]

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
        typelist = list(typelist)
        def cmp_bitwidth(ob1, ob2):
            ob1_bw_opt = getattr(ob1, 'bitwidth', ob1)
            ob2_bw_opt = getattr(ob2, 'bitwidth', ob2)
            return cmp(ob1_bw_opt, ob2_bw_opt)
        typelist.sort(cmp_bitwidth)
        return functools.reduce(self.unify_pairs, typelist)

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


