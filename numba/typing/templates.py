"""
Define typing templates
"""
from __future__ import print_function, division, absolute_import

import functools
from .. import types
from ..typeinfer import TypingError
from functools import reduce
import operator


class Signature(object):
    __slots__ = 'return_type', 'args', 'recvr'

    def __init__(self, return_type, args, recvr):
        self.return_type = return_type
        self.args = args
        self.recvr = recvr

    def __getstate__(self):
        """
        Needed because of __slots__.
        """
        return self.return_type, self.args, self.recvr

    def __setstate__(self, state):
        """
        Needed because of __slots__.
        """
        self.return_type, self.args, self.recvr = state

    def __hash__(self):
        return hash(self.args)

    def __eq__(self, other):
        if isinstance(other, Signature):
            return (self.args == other.args and
                    self.recvr == other.recvr)

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return "%s -> %s" % (self.args, self.return_type)

    @property
    def is_method(self):
        return self.recvr is not None


def make_concrete_template(name, key, signatures):
    baseclasses = (ConcreteTemplate,)
    gvars = dict(key=key, cases=list(signatures))
    return type(name, baseclasses, gvars)


def signature(return_type, *args, **kws):
    recvr = kws.pop('recvr', None)
    assert not kws
    return Signature(return_type, args, recvr=recvr)


def _uses_downcast(dists):
    for d in dists:
        if d < 0:
            return True
    return False


def _sum_downcast(dists):
    c = 0
    for d in dists:
        if d < 0:
            c += abs(d)
    return c


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


def _rate_arguments(context, actualargs, formalargs):
    ratings = [Rating()]
    for actual, formal in zip(actualargs, formalargs):
        rate = Rating()
        by = context.type_compatibility(actual, formal)
        if by is None:
            return None

        if by == 'promote':
            rate.promote += 1
        elif by == 'safe':
            rate.safe_convert += 1
        elif by == 'unsafe':
            rate.unsafe_convert += 1
        elif by == 'exact':
            pass
        else:
            raise Exception("unreachable", by)

        ratings.append(rate)
    return ratings


def resolve_overload(context, key, cases, args, kws):
    assert not kws, "Keyword arguments are not supported, yet"
    # Rate each cases
    candids = []
    symm_ratings = []
    for case in cases:
        if len(args) == len(case.args):
            ratings = _rate_arguments(context, args, case.args)
            if ratings is not None:
                combined = reduce(operator.add, ratings)
                symm_ratings.append(combined.astuple())
                candids.append(case)

    # Find the best case
    ordered = sorted(zip(symm_ratings, candids), key=lambda i: i[0])
    if ordered:
        if len(ordered) > 1:
            (first, case1), (second, case2) = ordered[:2]
            # Ambiguous overloading
            # NOTE: we can have duplicate overloadings if e.g. some type
            # aliases were used when declaring the supported signatures
            # (typical example being "intp" and "int64" on a 64-bit build)
            if first == second and case1 != case2:
                ambiguous = []
                for rate, case in ordered:
                    if rate == first:
                        ambiguous.append(case)

                # Try to resolve promotion
                # TODO: need to match this to the C overloading dispatcher
                resolvable = resolve_ambiguous_resolution(context, ambiguous,
                                                          args)
                if resolvable:
                    return resolvable

                # Failed to resolve promotion
                args = (key, args, '\n'.join(map(str, ambiguous)))
                msg = "Ambiguous overloading for %s %s\n%s" % args
                raise TypeError(msg)

        return ordered[0][1]


def resolve_ambiguous_resolution(context, cases, args):
    """Uses asymmetric resolution to find the best version
    """
    ratings = []
    for case in cases:
        rates = _rate_arguments(context, args, case.args)
        ratings.append((tuple(r.astuple() for r in rates), case))

    return max(ratings, key=lambda x: x[0])[1]


class FunctionTemplate(object):
    def __init__(self, context):
        self.context = context

    def _select(self, cases, args, kws):
        selected = resolve_overload(self.context, self.key, cases, args, kws)
        return selected


class AbstractTemplate(FunctionTemplate):
    """
    Defines method ``generic(self, args, kws)`` which compute a possible
    signature base on input types.  The signature does not have to match the
    input types. It is compared against the input types afterwards.
    """

    def apply(self, args, kws):
        generic = getattr(self, "generic")
        sig = generic(args, kws)

        # Unpack optional type if no matching signature
        if not sig and any(isinstance(x, types.Optional) for x in args):
            def unpack_opt(x):
                if isinstance(x, types.Optional):
                    return x.type
                else:
                    return x

            args = list(map(unpack_opt, args))
            assert not kws  # Not supported yet
            sig = generic(args, kws)

        if sig:
            cases = [sig]
            return self._select(cases, args, kws)


class ConcreteTemplate(FunctionTemplate):
    """
    Defines attributes "cases" as a list of signature to match against the
    given input types.
    """

    def apply(self, args, kws):
        cases = getattr(self, 'cases')
        assert cases
        return self._select(cases, args, kws)


class UntypedAttributeError(TypingError):
    def __init__(self, value, attr):
        msg = 'Unknown attribute "{attr}" of type {type}'.format(type=value,
                                                              attr=attr)
        super(UntypedAttributeError, self).__init__(msg)


class AttributeTemplate(object):
    def __init__(self, context):
        self.context = context

    def resolve(self, value, attr):
        ret = self._resolve(value, attr)
        if ret is None:
            raise UntypedAttributeError(value=value, attr=attr)
        return ret

    def _resolve(self, value, attr):
        fn = getattr(self, "resolve_%s" % attr, None)
        if fn is None:
            fn = self.generic_resolve
            if fn is NotImplemented:
                return self.context.resolve_module_constants(value, attr)
            else:
                return fn(value, attr)
        else:
            return fn(value)

    generic_resolve = NotImplemented


def bound_function(template_key):
    """
    Wrap an AttributeTemplate resolve_* method to allow it to
    resolve an instance method's signature rather than a instance attribute.
    The wrapped method must return the resolved method's signature
    according to the given self type, args, and keywords.

    It is used thusly:

        class ComplexAttributes(AttributeTemplate):
            @bound_function("complex.conjugate")
            def resolve_conjugate(self, ty, args, kwds):
                return ty

    *template_key* (e.g. "complex.conjugate" above) will be used by the
    target to look up the method's implementation, as a regular function.
    """
    def wrapper(method_resolver):
        @functools.wraps(method_resolver)
        def attribute_resolver(self, ty):
            class MethodTemplate(AbstractTemplate):
                key = template_key
                def generic(_, args, kws):
                    sig = method_resolver(self, ty, args, kws)
                    if sig is not None:
                        sig.recvr = ty
                        return sig

            return types.BoundFunction(MethodTemplate, ty)
        return attribute_resolver
    return wrapper


class ClassAttrTemplate(AttributeTemplate):
    def __init__(self, context, key, clsdict):
        super(ClassAttrTemplate, self).__init__(context)
        self.key = key
        self.clsdict = clsdict

    def resolve(self, value, attr):
        return self.clsdict[attr]


class MacroTemplate(object):
    pass


# -----------------------------

class Registry(object):
    def __init__(self):
        self.functions = []
        self.attributes = []
        self.globals = []

    def register(self, item):
        assert issubclass(item, FunctionTemplate)
        self.functions.append(item)
        return item

    def register_attr(self, item):
        assert issubclass(item, AttributeTemplate)
        self.attributes.append(item)
        return item

    def register_global(self, v, t):
        self.globals.append((v, t))

    def resolves_global(self, global_value, wrapper_type=types.Function):
        """
        Decorate a FunctionTemplate subclass so that it gets registered
        as resolving *global_value* with the *wrapper_type* (by default
        a types.Function).

        Example use::
            @resolves_global(math.fabs)
            class Math(ConcreteTemplate):
                cases = [signature(types.float64, types.float64)]
        """
        def decorate(cls):
            class Template(cls):
                key = global_value
            self.register_global(global_value, wrapper_type(Template))
            return cls
        return decorate


builtin_registry = Registry()
builtin = builtin_registry.register
builtin_attr = builtin_registry.register_attr
builtin_global = builtin_registry.register_global
