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
    __slots__ = 'funcobj', 'args', 'return_type'
    
    def __init__(self, funcobj, args, return_type):
        self.funcobj = funcobj
        self.args = tuple(x for x in args)
        self.return_type = return_type

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
        self.lib = defaultdict(set)

    def define(self, func):
        bin = self.lib[func.funcobj]
        if func in bin:
            raise ValueError('duplicated function')
        bin.add(func)

    def lookup(self, func, args):
        versions = self.lib[func]
        for ver in versions:
            if ver.args == args:
                return ver

    def get(self, func, args):
        graded = []
        versions = self.lib[func]

        if not versions:
            raise NotImplementedError('function %s is not implemented' % func)

        for ver in versions:
            if len(args) == len(ver.args):
                # reject coercion on first argument for instance attribute
                if (isinstance(func, str) and
                        func.startswith('.') and
                        args[0] != ver.args[0]):
                    continue
                # try to coerce to arguments and calculate a grade
                cur = tuple(actual.try_coerce(formal)
                            for actual, formal in zip(args, ver.args))

                if not any(x is None for x in cur):
                    graded.append((cur, ver))

        # filter out downcast
        no_downcast = [(sum(pts), defn) for pts, defn in graded
                       if not any(p < 0 for p in pts)]

        least_promotion = sorted(no_downcast)
        if not least_promotion:
            msg = 'no matching definition for %s(%s)'
            raise TypeError(msg % (func, ', '.join(str(a) for a in args)))
        return least_promotion[0][1]


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

