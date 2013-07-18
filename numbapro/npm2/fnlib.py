from collections import defaultdict
import operator
from .types import (int8, int16, int32, int64, intp,
                    uint8, uint16, uint32, uint64,
                    float32, float64,
                    complex64, complex128,
                    boolean, range_type, range_iter_type)
from . import types
from .typesets import (signed_set, unsigned_set, integer_set, float_set,
                       complex_set)

class Function(object):
    __slots__ = 'funcobj', 'args', 'return_type', 'is_parametric'
    
    def __init__(self, funcobj, args, return_type):
        self.funcobj = funcobj
        self.args = tuple(x for x in args)
        self.return_type = return_type
        self.is_parametric = any(isinstance(a, types.Kind) for a in args)

    def __hash__(self):
        return hash((self.funcobj, self.args))

    def __eq__(self, other):
        return (self.funcobj is other.funcobj and self.args == other.args)

    def __repr__(self):
        return '%s :: (%s) -> %s' % (self.funcobj,
                               ', '.join(str(a) for a in self.args),
                               self.return_type)


class FunctionLibrary(object):
    def __init__(self):
        self.concrete = defaultdict(set)
        self.parametric = defaultdict(set)

    def define(self, func):
        if func.is_parametric:
            bin = self.parametric[func.funcobj]
        else:
            bin = self.concrete[func.funcobj]

        if func in bin:
            raise ValueError('duplicated function')
        bin.add(func)

    def lookup(self, func, args):
        versions = self.parametric.get(func)
        if versions is None:
            versions = self.concrete[func]
        for ver in versions:
            if ver.args == args:
                return ver

    def get(self, func, args):
        result = self.get_parametric(func, args)
        if result is None:
            result = self.get_concrete(func, args)
        return result

    def get_parametric(self, func, args):
        if func not in self.parametric:
            return
        versions = self.parametric[func]

        accepted = []
        for ver in versions:
            matched = self._match_parametric_args(args, ver.args)
            if matched is not None:
                accepted.append((matched, ver))

        if accepted:
            least_promotion = sorted(accepted)
            return self._setup_param_defn(accepted[0][1], args)

    def _setup_param_defn(self, defn, actual_params):
        if callable(defn.return_type):
            return_type = defn.return_type(actual_params)
            return Function(funcobj     = defn.funcobj,
                            args        = defn.args,
                            return_type = return_type)
        return defn

    def _match_parametric_args(self, actual_params, formal_params):
        if len(actual_params) != len(formal_params):
            return

        pts = []
        for actual, formal in zip(actual_params, formal_params):
            if isinstance(formal, types.Kind):
                if not formal.matches(actual):
                    return
                pts.append(0)
            else:
                pt = actual.try_coerce(formal)
                if pt is None:
                    return
                if pt < 0:      # no downcast
                    return
                pts.append(pt)

        return sum(pts)

    def get_concrete(self, func, args):
        versions = self.concrete[func]

        if not versions:
            raise NotImplementedError('function %s is not implemented' % func)

        accepted = []
        for ver in versions:
            m = self._match_concrete_args(func, args, ver.args)
            if m is not None:
                accepted.append((m, ver))

        least_promotion = sorted(accepted)
        if not least_promotion:
            msg = 'no matching definition for %s(%s)'
            raise TypeError(msg % (func, ', '.join(str(a) for a in args)))
        return least_promotion[0][1]

    def _match_concrete_args(self, func, actual_params, formal_params):
        # num of args must match
        if len(actual_params) != len(formal_params):
            return

        # reject coercion for the first argument for instance attribute
        if (isinstance(func, str) and func.startswith('.') and
                    actual_params[0] != formal_params[0]):
            return

        # try promotion
        pts = []
        for actual, formal in zip(actual_params, formal_params):
            pt = actual.try_coerce(formal)
            if pt is None:
                return
            if pt < 0:      # no downcast
                return
            pts.append(pt)

        return sum(pts)

#------------------------------------------------------------------------------
# builtin functions

def binary_func_from_sets(typesets):
    return [binary_func(t) for t in typesets]

def binary_func(ty):
    return (ty, ty), ty

def binary_div(ty, out):
    return [((ty, ty), out)]

def floor_divisions():
    out = []
    for ty in integer_set:
        out += binary_div(ty, ty)
    out += binary_div(types.float32, types.int32)
    out += binary_div(types.float64, types.int64)
    return out

def bool_func_from_sets(typesets):
    return [bool_func(t) for t in typesets]

def bool_func(ty):
    return (ty, ty), boolean

def unary_func_from_sets(typesets):
    return [unary_func(t) for t in typesets]

def unary_func(ty):
    return (ty,), ty


def range_func():
    return [((intp, intp, intp),    range_type),
            ((intp, intp),          range_type),
            ((intp,),               range_type)]

def iter_func():
    return [((range_type,), range_iter_type)]

def iter_valid_func():
    return [((range_iter_type,), boolean)]

def iter_next_func():
    return [((range_iter_type,), intp)]

def complex_attr(typeset):
    return [((ty,), ty.desc.element)
            for ty in typeset]

def complex_ctor(typeset):
    defns = []
    for ty in typeset:
        elem = ty.desc.element
        ver1 = ((elem,), ty)
        ver2 = ((elem, elem), ty)
        defns.append(ver1)
        defns.append(ver2)
    return defns

def int_ctor(typeset):
    defns = []
    for ty in typeset:
        defn = ((ty,), types.intp)
        defns.append(defn)
    return defns

def float_ctor(typeset):
    defns = []
    for ty in typeset:
        defn = ((ty,), types.float64)
        defns.append(defn)
    return defns

def array_getitem_return(args):
    ary, idx = args
    return ary.desc.element

builtins = {
    range           : range_func(),
    xrange          : range_func(),
    iter            : iter_func(),
    'itervalid'     : iter_valid_func(),
    'iternext'      : iter_next_func(),

    operator.add: binary_func_from_sets(integer_set|float_set|complex_set),
    operator.sub: binary_func_from_sets(integer_set|float_set|complex_set),
    operator.mul: binary_func_from_sets(integer_set|float_set|complex_set),
    operator.floordiv: floor_divisions(),
    operator.truediv: binary_func_from_sets(float_set),
    operator.mod: binary_func_from_sets(integer_set|float_set),

    operator.lshift: binary_func_from_sets(integer_set),
    operator.rshift: binary_func_from_sets(integer_set),

    operator.and_: binary_func_from_sets(integer_set),
    operator.or_: binary_func_from_sets(integer_set),
    operator.xor: binary_func_from_sets(integer_set),

    operator.neg: unary_func_from_sets(signed_set|float_set|complex_set),
    operator.invert: unary_func_from_sets(integer_set),

    operator.gt: bool_func_from_sets(integer_set|float_set|complex_set),
    operator.lt: bool_func_from_sets(integer_set|float_set|complex_set),
    operator.ge: bool_func_from_sets(integer_set|float_set|complex_set),
    operator.le: bool_func_from_sets(integer_set|float_set|complex_set),
    operator.eq: bool_func_from_sets(integer_set|float_set|complex_set),
    operator.ne: bool_func_from_sets(integer_set|float_set|complex_set),

    '.real': complex_attr(complex_set),
    '.imag': complex_attr(complex_set),
    complex: complex_ctor(complex_set),

    int:   int_ctor(integer_set|float_set|complex_set),
    float: float_ctor(integer_set|float_set|complex_set),

    operator.getitem: [((types.ArrayKind, types.intp),
                            array_getitem_return),
                       ((types.ArrayKind, types.TupleKind),
                            array_getitem_return),]
}

def get_builtin_function_library(lib=None):
    '''Create or add builtin functions to a FunctionLibrary instance.
    '''
    lib = FunctionLibrary() if lib is None else lib
    for k, vl in builtins.iteritems():
        for args, return_type in vl:
            fn = Function(funcobj=k, args=args, return_type=return_type)            
            lib.define(fn)
    return lib

