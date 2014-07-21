"""
Define typing templates
"""
from __future__ import print_function, division, absolute_import
from numba import types


class Signature(object):
    __slots__ = 'return_type', 'args', 'recvr'

    def __init__(self, return_type, args, recvr):
        self.return_type = return_type
        self.args = args
        self.recvr = recvr

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


def resolve_overload(context, key, cases, args, kws):
    assert not kws, "Keyword arguments are not supported, yet"
    # Rate each cases
    candids = []
    ratings = []
    for case in cases:
        if len(args) == len(case.args):
            rate = Rating()
            for actual, formal in zip(args, case.args):
                by = context.type_compatibility(actual, formal)
                if by is None:
                    break

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

            else:
                ratings.append(rate.astuple())
                candids.append(case)

    # Find the best case
    ordered = sorted(zip(ratings, candids), key=lambda i: i[0])
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
                resolvable = resolve_ambiguous_promotions(context, ambiguous,
                                                          args)
                if resolvable:
                    return resolvable

                # Failed to resolve promotion
                args = (key, args, '\n'.join(map(str, ambiguous)))
                msg = "Ambiguous overloading for %s %s\n%s" % args
                raise TypeError(msg)

        return ordered[0][1]


class UnsafePromotionError(Exception):
    pass


def safe_promotion(actual, formal):
    """
    Allow integer to be casted to the nearest integer
    """
    # Integers?
    if actual in types.integer_domain and formal in types.integer_domain:
        # Same signedness?
        if actual.signed == formal.signed:
            # Score by their distance
            return formal.bitwidth - actual.bitwidth
    raise UnsafePromotionError(actual, formal)


def resolve_ambiguous_promotions(context, cases, args):
    ratings = []
    for case in cases:
        try:
            rate = _safe_promote_case(context, case, args)
        except UnsafePromotionError:
            # Ignore error
            pass
        else:
            ratings.append((rate, case))

    _, bestcase = min(ratings)
    return bestcase


def _safe_promote_case(context, case, args):
    rate = 0
    for actual, formal in zip(args, case.args):
        by = context.type_compatibility(actual, formal)
        if by == 'promote':
            rate += safe_promotion(actual, formal)
        else:
            raise UnsafePromotionError(actual, formal)
    return rate


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


class AttributeTemplate(object):
    def __init__(self, context):
        self.context = context

    def resolve(self, value, attr):
        fn = getattr(self, "resolve_%s" % attr, None)
        if fn is None:
            fn = self.generic_resolve
            if fn is NotImplemented:
                attrty = self.context.resolve_module_constants(value, attr)
                if attrty is not None:
                    return attrty
                raise NotImplementedError(value, attr)
            else:
                return fn(value, attr)
        else:
            return fn(value)

    generic_resolve = NotImplemented


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


builtin_registry = Registry()
builtin = builtin_registry.register
builtin_attr = builtin_registry.register_attr
builtin_global = builtin_registry.register_global
