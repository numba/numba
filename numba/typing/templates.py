"""
Define typing templates
"""
from __future__ import print_function
import math
import numpy
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
builtin_global(slice, types.slice_type)
builtin_global(abs, types.abs_type)
builtin_global(print, types.print_type)


@builtin
class Print(ConcreteTemplate):
    key = types.print_type
    intcases = [signature(types.none, ty) for ty in types.integer_domain]
    realcases = [signature(types.none, ty) for ty in types.real_domain]
    cases = intcases + realcases


@builtin
class Abs(ConcreteTemplate):
    key = types.abs_type
    cases = [signature(ty, ty) for ty in types.signed_domain]


@builtin
class Slice(ConcreteTemplate):
    key = types.slice_type
    cases = [
        signature(types.slice2_type, types.intp, types.intp),
        signature(types.slice3_type, types.intp, types.intp, types.intp),
    ]


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
        signature(types.uint8, types.uint8, types.uint8),
        signature(types.uint16, types.uint16, types.uint16),
        signature(types.uint32, types.uint32, types.uint32),
        signature(types.uint64, types.uint64, types.uint64),

        signature(types.int8, types.int8, types.int8),
        signature(types.int16, types.int16, types.int16),
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


@builtin
class BinOpMod(BinOp):
    key = "%"


@builtin
class BinOpPower(ConcreteTemplate):
    key = "**"
    cases = [
        signature(types.float64, types.float64, types.int32),
        signature(types.float32, types.float32, types.float32),
        signature(types.float64, types.float64, types.float64),
    ]

class CmpOp(ConcreteTemplate):
    cases = [
        signature(types.boolean, types.uint8, types.uint8),
        signature(types.boolean, types.uint16, types.uint16),
        signature(types.boolean, types.uint32, types.uint32),
        signature(types.boolean, types.uint64, types.uint64),

        signature(types.boolean, types.int8, types.int8),
        signature(types.boolean, types.int16, types.int16),
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

    elif index == types.slice3_type:
        return types.slice3_type

    elif index == types.slice2_type:
        return types.slice2_type

    else:
        return types.intp


@builtin
class GetItemUniTuple(AbstractTemplate):
    key = "getitem"

    def generic(self, args, kws):
        tup, idx = args
        if isinstance(tup, types.UniTuple):
            return signature(tup.dtype, tup, normalize_index(idx))


@builtin
class GetItemArray(AbstractTemplate):
    key = "getitem"

    def generic(self, args, kws):
        assert not kws
        ary, idx = args
        if not isinstance(ary, types.Array):
            return

        idx = normalize_index(idx)
        if idx in (types.slice2_type, types.slice3_type):
            res = ary.copy(layout='A')
        elif isinstance(idx, types.UniTuple):
            if ary.ndim > len(idx):
                return
            elif ary.ndim < len(idx):
                return
            else:
                res = ary.dtype
        elif idx == types.intp:
            if ary.ndim != 1:
                return
            res = ary.dtype
        else:
            raise Exception("unreachable")

        return signature(res, ary, idx)


@builtin
class SetItemArray(AbstractTemplate):
    key = "setitem"

    def generic(self, args, kws):
        assert not kws
        ary, idx, val = args
        if isinstance(ary, types.Array):
            return signature(types.none, ary, normalize_index(idx), ary.dtype)


@builtin
class LenArray(AbstractTemplate):
    key = types.len_type

    def generic(self, args, kws):
        assert not kws
        (ary,) = args
        if isinstance(ary, types.Array):
            return signature(types.intp, ary)

#-------------------------------------------------------------------------------

@builtin
class ArrayAttribute(AttributeTemplate):
    key = types.Array

    def resolve_shape(self, ary):
        return types.UniTuple(types.intp, ary.ndim)

    def resolve_flatten(self, ary):
        return types.Method(Array_flatten, ary)


class Array_flatten(AbstractTemplate):
    key = "array.flatten"

    def generic(self, args, kws):
        assert not args
        assert not kws
        this = self.this
        if this.layout == 'C':
            resty = this.copy(ndim=1)
            return signature(resty, recvr=this)


@builtin
class CmpOpEqArray(AbstractTemplate):
    key = '=='

    def generic(self, args, kws):
        assert not kws
        [va, vb] = args
        if isinstance(va, types.Array) and va == vb:
            return signature(va.copy(dtype=types.boolean), va, vb)

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

#-------------------------------------------------------------------------------

@builtin
class NumpyModuleAttribute(AttributeTemplate):
    key = types.Module(numpy)

    def resolve_absolute(self, mod):
        return types.Function(Numpy_absolute)

    def resolve_exp(self, mod):
        return types.Function(Numpy_exp)

    def resolve_sin(self, mod):
        return types.Function(Numpy_sin)

    def resolve_cos(self, mod):
        return types.Function(Numpy_cos)

    def resolve_tan(self, mod):
        return types.Function(Numpy_tan)

    def resolve_add(self, mod):
        return types.Function(Numpy_add)

    def resolve_subtract(self, mod):
        return types.Function(Numpy_subtract)

    def resolve_multiply(self, mod):
        return types.Function(Numpy_multiply)

    def resolve_divide(self, mod):
        return types.Function(Numpy_divide)


class Numpy_unary_ufunc(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        [inp, out] = args
        if isinstance(inp, types.Array) and isinstance(out, types.Array):
            if inp.dtype != out.dtype:
                # TODO handle differing dtypes
                return
            return signature(out, inp, out)


class Numpy_absolute(Numpy_unary_ufunc):
    key = numpy.absolute


class Numpy_sin(Numpy_unary_ufunc):
    key = numpy.sin


class Numpy_cos(Numpy_unary_ufunc):
    key = numpy.cos


class Numpy_tan(Numpy_unary_ufunc):
    key = numpy.tan


class Numpy_exp(Numpy_unary_ufunc):
    key = numpy.exp


class Numpy_binary_ufunc(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        [vx, wy, out] = args
        if (isinstance(vx, types.Array) and isinstance(wy, types.Array) and
                isinstance(out, types.Array)):
            if vx.dtype != wy.dtype and vx.dtype != out.dtype:
                # TODO handle differing dtypes
                return
            return signature(out, vx, wy, out)


class Numpy_add(Numpy_binary_ufunc):
    key = numpy.add


class Numpy_subtract(Numpy_binary_ufunc):
    key = numpy.subtract


class Numpy_multiply(Numpy_binary_ufunc):
    key = numpy.multiply


class Numpy_divide(Numpy_binary_ufunc):
    key = numpy.divide


builtin_global(numpy, types.Module(numpy))
builtin_global(numpy.absolute, types.Function(Numpy_absolute))
builtin_global(numpy.exp, types.Function(Numpy_exp))
builtin_global(numpy.sin, types.Function(Numpy_sin))
builtin_global(numpy.cos, types.Function(Numpy_cos))
builtin_global(numpy.tan, types.Function(Numpy_tan))
builtin_global(numpy.add, types.Function(Numpy_add))
builtin_global(numpy.subtract, types.Function(Numpy_subtract))
builtin_global(numpy.multiply, types.Function(Numpy_multiply))
builtin_global(numpy.divide, types.Function(Numpy_divide))


