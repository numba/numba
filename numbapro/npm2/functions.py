from collections import defaultdict
import operator
from .types import (int8, int16, int32, int64, intp,
                    uint8, uint16, uint32, uint64,
                    float32, float64,
                    complex64, complex128,
                    boolean,
                    RangeType, RangeIterType)
from .typesets import (signed_set, unsigned_set, integer_set, float_set,
                       complex_set)

class Function(object):
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

    def get(self, func, args):
        graded = []
        versions = self.lib[func]

        if not versions:
            raise NotImplementedError('function %s is not implemented' % func)
        for ver in versions:
            if len(args) == len(ver.args):
                cur = tuple(actual.coerce(formal, noraise=True)
                            for actual, formal in zip(args, ver.args))
                graded.append((cur, ver))

        # filter out downcast
        no_downcast = [(sum(pts), defn) for pts, defn in graded
                       if not any(p < 0 for p in pts)]


        least_promotion = sorted(no_downcast)
        return least_promotion[0][1]


#------------------------------------------------------------------------------
# builtin functions

def binary_func_from_sets(typesets):
    return [binary_func(t) for t in typesets]

def binary_func(ty):
    return (ty, ty), ty

def bool_func_from_sets(typesets):
    return [bool_func(t) for t in typesets]

def bool_func(ty):
    return (ty, ty), boolean

def range_func():
    return [((intp, intp, intp),    RangeType),
            ((intp, intp),          RangeType),
            ((intp,),               RangeType)]

def iter_func():
    return [((RangeType,), RangeIterType)]

def iter_valid_func():
    return [((RangeIterType,), boolean)]

def iter_next_func():
    return [((RangeIterType,), intp)]

builtins = {
    range           : range_func(),
    xrange          : range_func(),
    iter            : iter_func(),
    'itervalid'     : iter_valid_func(),
    'iternext'      : iter_next_func(),

    operator.add: binary_func_from_sets(integer_set|float_set|complex_set),
    operator.sub: binary_func_from_sets(integer_set|float_set|complex_set),
    operator.mul: binary_func_from_sets(integer_set|float_set|complex_set),
    operator.div: binary_func_from_sets(integer_set|float_set|complex_set),
    
    operator.eq: bool_func_from_sets(integer_set|float_set|complex_set),
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

