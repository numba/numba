"""
Define typing templates
"""
import math
from numba import types


class Signature(object):
    __slots__ = 'return_type', 'args'

    def __init__(self, return_type, args):
        self.return_type = return_type
        self.args = args

    def __hash__(self):
        return hash(self.args)

    def __eq__(self, other):
        if isinstance(other, Signature):
            return self.args == other.args

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return "%s -> %s" % (self.args, self.return_type)


def signature(return_type, *args):
    return Signature(return_type, args)


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


class FunctionTemplate(object):
    def __init__(self, context):
        self.context = context

    def apply_case(self, case, args, kws):
        """
        Returns a tuple of type distances for each arguments
        or return None if not match.
        """
        assert not kws, "Keyword argument is not supported, yet"
        if len(case.args) != len(args):
            # Number of arguments mismatch
            return None
        distances = []
        for formal, actual in zip(case.args, args):
            tdist = self.context.type_distance(toty=formal, fromty=actual)
            if tdist is None:
                return
            distances.append(tdist)
        return tuple(distances)

    def _select(self, cases, args, kws):
        upcast, downcast = self._find_compatible_definitions(cases, args, kws)
        return self._select_best_definition(upcast, downcast, args, kws,
                                            cases)

    def _find_compatible_definitions(self, cases, args, kws):
        upcast = []
        downcast = []
        for case in cases:
            dists = self.apply_case(case, args, kws)
            if dists is not None:
                if _uses_downcast(dists):
                    downcast.append((dists, case))
                else:
                    upcast.append((sum(dists), case))
        return upcast, downcast

    def _select_best_definition(self, upcast, downcast, args, kws, cases):
        if upcast:
            return self._select_best_upcast(upcast)
        elif downcast:
            return self._select_best_downcast(downcast)

    def _select_best_downcast(self, downcast):
        assert downcast
        if len(downcast) == 1:
            # Exactly one definition with downcasting
            return downcast[0][1]
        else:
            downdist = sys.maxint
            leasts = []
            for dists, case in downcast:
                n = _sum_downcast(dists)
                if n < downdist:
                    downdist = n
                    leasts = [(dists, case)]
                elif n == downdist:
                    leasts.append((dists, case))

            if len(leasts) == 1:
                return leasts[0][1]
            else:
                # Need to further decide which downcasted version?
                raise TypeError("Ambiguous overloading: %s" %
                                [c for _, c in leasts])

    def _select_best_upcast(self, upcast):
        assert upcast
        if len(upcast) == 1:
            # Exactly one definition without downcasting
            return upcast[0][1]
        else:
            assert len(upcast) > 1
            first = min(upcast)
            upcast.remove(first)
            second = min(upcast)
            if first[0] < second[0]:
                return first[1]
            else:
                raise TypeError("Ambiguous overloading: %s and %s" % (
                    first[1], second[1]))


class AbstractTemplate(FunctionTemplate):
    """
    Defines method ``generic(self, args, kws)`` which compute a possible
    signature base on input types.  The signature does not have to match the
    input types. It is compared against the input types afterwards.
    """

    def apply(self, args, kws):
        generic = getattr(self, "generic")
        cases = [generic(args, kws)]
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
            raise NotImplementedError(value, attr)
        return fn(value)


class ClassAttrTemplate(AttributeTemplate):
    def __init__(self, context, key, clsdict):
        super(ClassAttrTemplate, self).__init__(context)
        self.key = key
        self.clsdict = clsdict

    def resolve(self, value, attr):
        return self.clsdict[attr]


# -----------------------------------------------------------------------------

BUILTINS = []
BUILTIN_ATTRS = []
BUILTIN_GLOBALS = []


def builtin(template):
    if issubclass(template, AttributeTemplate):
        BUILTIN_ATTRS.append(template)
    else:
        BUILTINS.append(template)
    return template


def builtin_global(v, t):
    BUILTIN_GLOBALS.append((v, t))


builtin_global(range, types.range_type)
builtin_global(xrange, types.range_type)
builtin_global(len, types.len_type)


@builtin
class Range(ConcreteTemplate):
    key = types.range_type
    cases = [
        signature(types.range_state32_type, types.int32),
        signature(types.range_state32_type, types.int32, types.int32),
        signature(types.range_state64_type, types.int64),
        signature(types.range_state64_type, types.int64, types.int64),
    ]


@builtin
class GetIter(ConcreteTemplate):
    key = "getiter"
    cases = [
        signature(types.range_iter32_type, types.range_state32_type),
        signature(types.range_iter64_type, types.range_state64_type),
    ]


@builtin
class IterNext(ConcreteTemplate):
    key = "iternext"
    cases = [
        signature(types.int32, types.range_iter32_type),
        signature(types.int64, types.range_iter64_type),
    ]


@builtin
class IterValid(ConcreteTemplate):
    key = "itervalid"
    cases = [
        signature(types.boolean, types.range_iter32_type),
        signature(types.boolean, types.range_iter64_type),
    ]


class BinOp(ConcreteTemplate):
    cases = [
        signature(types.int32, types.int32, types.int32),
        signature(types.int64, types.int64, types.int64),
        signature(types.float32, types.float32, types.float32),
        signature(types.float64, types.float64, types.float64),
    ]


@builtin
class BinOpAdd(BinOp):
    key = "+"


@builtin
class BinOpSub(BinOp):
    key = "-"


@builtin
class BinOpMul(BinOp):
    key = "*"


@builtin
class BinOpDiv(BinOp):
    key = "/?"


class CmpOp(ConcreteTemplate):
    cases = [
        signature(types.boolean, types.int32, types.int32),
        signature(types.boolean, types.int64, types.int64),
        signature(types.boolean, types.float32, types.float32),
        signature(types.boolean, types.float64, types.float64),
    ]


@builtin
class CmpOpLt(CmpOp):
    key = '<'


@builtin
class CmpOpLe(CmpOp):
    key = '<='


@builtin
class CmpOpGt(CmpOp):
    key = '>'


@builtin
class CmpOpGe(CmpOp):
    key = '>='


@builtin
class CmpOpEq(CmpOp):
    key = '=='


@builtin
class CmpOpNe(CmpOp):
    key = '!='


def normalize_index(index):
    if isinstance(index, types.UniTuple):
        return types.UniTuple(types.intp, index.count)

    else:
        return types.intp


@builtin
class GetItemUniTuple(AbstractTemplate):
    key = "getitem"
    object = types.Kind(types.UniTuple)

    def generic(self, args, kws):
        assert not kws
        tup, idx = args
        return signature(tup.dtype, tup, normalize_index(idx))


@builtin
class GetItemArray(AbstractTemplate):
    key = "getitem"

    def generic(self, args, kws):
        assert not kws
        ary, idx = args
        return signature(ary.dtype, ary, normalize_index(idx))


@builtin
class SetItemArray(AbstractTemplate):
    key = "setitem"

    def generic(self, args, kws):
        assert not kws
        ary, idx, val = args
        return signature(types.none, ary, normalize_index(idx), ary.dtype)


@builtin
class LenArray(AbstractTemplate):
    key = types.len_type

    def generic(self, args, kws):
        assert not kws
        (ary,) = args
        return signature(types.intp, ary)

#-------------------------------------------------------------------------------

@builtin
class ArrayAttribute(AttributeTemplate):
    key = types.Array

    def resolve_shape(self, ary):
        return types.UniTuple(types.intp, ary.ndim)

#-------------------------------------------------------------------------------

@builtin
class MathModuleAttribute(AttributeTemplate):
    key = types.Module(math)

    def resolve_fabs(self, mod):
        return types.Function(Math_fabs)

    def resolve_exp(self, mod):
        return types.Function(Math_exp)

    def resolve_sqrt(self, mod):
        return types.Function(Math_sqrt)

    def resolve_log(self, mod):
        return types.Function(Math_log)


class Math_fabs(ConcreteTemplate):
    key = math.fabs
    cases = [
        signature(types.float32, types.float32),
        signature(types.float64, types.float64),
    ]


class Math_exp(ConcreteTemplate):
    key = math.exp
    cases = [
        signature(types.float32, types.float32),
        signature(types.float64, types.float64),
    ]


class Math_sqrt(ConcreteTemplate):
    key = math.sqrt
    cases = [
        signature(types.float32, types.float32),
        signature(types.float64, types.float64),
    ]


class Math_log(ConcreteTemplate):
    key = math.log
    cases = [
        signature(types.float32, types.float32),
        signature(types.float64, types.float64),
    ]


builtin_global(math, types.Module(math))
builtin_global(math.fabs, types.Function(Math_fabs))
builtin_global(math.exp, types.Function(Math_exp))
builtin_global(math.sqrt, types.Function(Math_sqrt))
builtin_global(math.log, types.Function(Math_log))
