import sys
from collections import namedtuple
from numba import types


def _type_distance(domain, first, second):
    if first in domain and second in domain:
        return domain.index(first) - domain.index(second)


class Context(object):
    """A typing context for storing function typing constrain template.

    """

    def __init__(self, type_lattice=None):
        self.type_lattice = type_lattice or types.type_lattice
        self.functions = {}
        self.load_builtins()

    def resolve_function_type(self, func, args, kws):
        ft = self.functions[func]
        return ft.apply(args, kws)

    def load_builtins(self):
        for ftcls in BUILTINS:
            self.insert_function(ftcls(self))

    def insert_function(self, ft):
        key = ft.key
        assert key not in self.functions, "Duplicated function template"
        self.functions[key] = ft

    def type_distance(self, fromty, toty):
        if fromty == toty:
            return 0

        return self.type_lattice.get((fromty, toty))

    def unify_types(self, *types):
        return reduce(self.unify_pairs, types)

    def unify_pairs(self, first, second):
        """
        Choose PyObject type as the abstract if we fail to determine a concrete
        type.
        """
        d = self.type_distance(fromty=first, toty=second)
        if d is None:
            return types.pyobject
        elif d >= 0:
            # A promotion from first -> second
            return second
        else:
            # A demontion from first -> second
            return first


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
    """
    A function typing template
    """
    __slots__ = 'context'

    def __init__(self, context):
        self.context = context

    def apply(self, args, kws):
        cases = getattr(self, 'cases', None)
        if cases:
            upcast = []
            downcast = []
            # Find compatible definitions
            for case in cases:
                dists = self.apply_case(case, args, kws)
                if dists is not None:
                    if _uses_downcast(dists):
                        downcast.append((dists, case))
                    else:
                        upcast.append((sum(dists), case))
                # Find best definition
            if not upcast and not downcast:
                # No matching definition
                raise TypeError(self.key, args, kws, cases)
            elif len(upcast) == 1:
                # Exactly one definition without downcasting
                return upcast[0][1].return_type
            elif len(upcast) > 1:
                first = min(upcast)
                upcast.remove(first)
                second = min(upcast)
                if first[0] < second[0]:
                    return first[1].return_type
                else:
                    raise TypeError("Ambiguous overloading: %s and %s" % (
                        first[1], second[1]))
            elif len(downcast) == 1:
                # Exactly one definition with downcasting
                return downcast[0][1].return_type
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
                    return leasts[0][1].return_type
                else:
                    # Need to further decide which downcasted version?
                    raise TypeError("Ambiguous overloading: %s" %
                                    [c for _, c in leasts])

        else:
            # TODO: generic function template
            raise NotImplementedError

    def apply_case(self, case, args, kws):
        """
        Returns a tuple of type distances for each arguments
        or return None if not match.
        """
        assert not kws, "Keyword argument is not supported, yet"
        if len(case.args) != len(args):
            return None
        distances = []
        for formal, actual in zip(case.args, args):
            tdist = self.context.type_distance(toty=formal, fromty=actual)
            if tdist is None:
                return
            distances.append(tdist)
        return tuple(distances)


_signature = namedtuple('signature', ['return_type', 'args'])


def signature(return_type, *args):
    return _signature(return_type, args)


class Range(FunctionTemplate):
    key = types.range_type
    cases = [
        signature(types.range_state32_type, types.int32),
        signature(types.range_state32_type, types.int32, types.int32),
        signature(types.range_state64_type, types.int64),
        signature(types.range_state64_type, types.int64, types.int64),
    ]


class GetIter(FunctionTemplate):
    key = "getiter"
    cases = [
        signature(types.range_iter32_type, types.range_state32_type),
        signature(types.range_iter64_type, types.range_state64_type),
    ]


class IterNext(FunctionTemplate):
    key = "iternext"
    cases = [
        signature(types.int32, types.range_iter32_type),
        signature(types.int64, types.range_iter64_type),
    ]


class IterValid(FunctionTemplate):
    key = "itervalid"
    cases = [
        signature(types.boolean, types.range_iter32_type),
        signature(types.boolean, types.range_iter64_type),
    ]


class BinOp(FunctionTemplate):
    cases = [
        signature(types.int32, types.int32, types.int32),
        signature(types.int64, types.int64, types.int64),
        signature(types.float32, types.float32, types.float32),
        signature(types.float64, types.float64, types.float64),
    ]


class BinOpAdd(BinOp):
    key = "+"


class BinOpSub(BinOp):
    key = "-"


class BinOpMul(BinOp):
    key = "*"


class BinOpDiv(BinOp):
    key = "/?"


BUILTINS = [
    Range,
    GetIter,
    IterNext,
    IterValid,
    # binary operations
    BinOpAdd,
    BinOpMul,
]
