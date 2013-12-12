from collections import defaultdict
import operator
from .types import boolean
from . import types
from .typesets import (signed_set, integer_set, float_set, complex_set)

class Function(object):
    __slots__ = 'funcobj', 'args', 'return_type', 'is_parametric'
    
    def __init__(self, funcobj, args, return_type):
        self.funcobj = funcobj
        self.args = None if args is None else tuple(x for x in args)
        self.return_type = return_type
        self.is_parametric = (args is not None and
                              (callable(return_type) or
                                    any(isinstance(a, types.Kind) or callable(a)
                                        for a in args)))

    def __hash__(self):
        return hash((self.funcobj, self.args))

    def __eq__(self, other):
        return (self.funcobj is other.funcobj and self.args == other.args)

    def __repr__(self):
        if self.args is None:
            return '%s :: (...) -> %s' % (self.funcobj, self.return_type)
        else:
            return '%s :: (%s) -> %s' % (self.funcobj,
                                         ', '.join(str(a) for a in self.args),
                                         self.return_type)

def _least_demontion(demontables):
    return sorted((sum(filter(lambda x: x < 0, pts)), ver)
                  for pts, ver in demontables)
    
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
            raise ValueError('duplicated function %s' % func)
        bin.add(func)

    def lookup(self, func, args):
        versions = self.parametric.get(func)
        if versions is None:
            versions = self.concrete[func]
        for ver in versions:
            if ver.args is None:
                return ver
            if len(ver.args) == len(args):
                for a, b in zip(ver.args, args):
                    if a != b:
                        break
                else:
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

        promotable = []
        demontable = []
        for ver in versions:
            pts = self._match_parametric_args(args, ver.args)
            if pts is not None:
                if all(x >= 0 for x in pts):
                    promotable.append((sum(pts), ver))
                else:
                    demontable.append((pts, ver))

        if promotable:
            least_promotion = sorted(promotable)
            return self._setup_param_defn(least_promotion[0][1], args)
        elif demontable:
            least_demotion = _least_demontion(demontable)
            return self._setup_param_defn(least_demotion[0][1], args)

    def _setup_param_defn(self, defn, actual_params):
        '''Return a new function object.
        If the return-type is parameteric, replace it with a concrete type.
        The new function object can be used lookup implementator.
        The return-type is never part of the signature.
        '''
        if callable(defn.return_type):
            return_type = defn.return_type(actual_params)
        else:
            return_type = defn.return_type

        return Function(funcobj     = defn.funcobj,
                        args        = defn.args,
                        return_type = return_type)

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
                if callable(formal):
                    formal = formal(actual_params)
                pt = (actual.try_coerce(formal)
                        if formal is not None
                        else None)
                if pt is None:
                    return
                pts.append(pt)

        return pts

    def get_concrete(self, func, args):
        versions = self.concrete[func]

        if not versions:
            raise NotImplementedError('function %s(%s) is not implemented' %
                                      (func, ', '.join(str(a) for a in args)))

        promotable = []
        demontable = []

        for ver in versions:
            pts = self._match_concrete_args(func, args, ver.args)
            if pts is not None:
                if all(p >= 0 for p in pts):
                    promotable.append((sum(pts), ver))
                else:
                    demontable.append((pts, ver))

        if promotable:
            least_promotion = sorted(promotable)
            return least_promotion[0][1]
        elif demontable:
            least_demotion = _least_demontion(demontable)
            return least_demotion[0][1]
        else:
            msg = 'no matching definition for %s(%s)'
            raise TypeError(msg % (func, ', '.join(str(a) for a in args)))


    def _match_concrete_args(self, func, actual_params, formal_params):
        if formal_params is None: # varargs
            return [0] * len(actual_params)

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
            pts.append(pt)

        return pts

#------------------------------------------------------------------------------
# builtin functions

def binary_func_from_sets(typesets):
    return [binary_func(t) for t in typesets]

def binary_func(ty):
    return (ty, ty), ty

def binary_div(ty, out):
    return [((ty, ty), out)]

def py2_divisions():
    out = []
    for ty in integer_set:
        out += binary_div(ty, ty)
    return out

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

def intparray_getitem_return(args):
    ary = args[0]
    return ary.desc.element

def minmax_from_sets(count, typeset):
    sigs = []
    for ty in typeset:
        sig = ((ty,) * count, ty)
        sigs.append(sig)
    return sigs

#-----------------------------------------------------------------------------
# Define Builtin Signatures

def def_(funcobj, definitions):
    return [(funcobj, definitions)]

builtins = []

builtins += def_(operator.add,
                   binary_func_from_sets(integer_set|float_set|complex_set))

builtins += def_(operator.sub,
                   binary_func_from_sets(integer_set|float_set|complex_set))

builtins += def_(operator.mul,
                   binary_func_from_sets(integer_set|float_set|complex_set))

builtins += def_(operator.div, py2_divisions())

builtins += def_(operator.div, binary_func_from_sets(float_set|complex_set))

builtins += def_(operator.floordiv, floor_divisions())

builtins += def_(operator.truediv, binary_func_from_sets(float_set|complex_set))

builtins += def_(operator.mod, binary_func_from_sets(integer_set|float_set))

builtins += def_(operator.lshift, binary_func_from_sets(integer_set))

builtins += def_(operator.rshift, binary_func_from_sets(integer_set))

builtins += def_(operator.and_, binary_func_from_sets(integer_set))

builtins += def_(operator.or_, binary_func_from_sets(integer_set))

builtins += def_(operator.xor, binary_func_from_sets(integer_set))

builtins += def_(operator.neg,
                   unary_func_from_sets(signed_set|float_set|complex_set))

builtins += def_(operator.invert, unary_func_from_sets(integer_set))

builtins += def_(operator.gt,
                   bool_func_from_sets(integer_set|float_set))

builtins += def_(operator.lt,
                   bool_func_from_sets(integer_set|float_set))

builtins += def_(operator.ge,
                   bool_func_from_sets(integer_set|float_set))

builtins += def_(operator.le,
                   bool_func_from_sets(integer_set|float_set))

builtins += def_(operator.eq,
                   bool_func_from_sets(integer_set|float_set|complex_set))

builtins += def_(operator.ne,
                   bool_func_from_sets(integer_set|float_set|complex_set))

builtins += def_('.real', complex_attr(complex_set))

builtins += def_('.imag', complex_attr(complex_set))

builtins += def_(complex, complex_ctor(complex_set))

builtins += def_(int, int_ctor(integer_set|float_set|complex_set))

builtins += def_(float, float_ctor(integer_set|float_set|complex_set))

builtins += def_(operator.getitem,
             [((types.FixedArrayKind, types.intp), intparray_getitem_return)])

builtins += def_(abs, unary_func_from_sets(integer_set|float_set))

builtins += def_(min, minmax_from_sets(2, integer_set|float_set))
builtins += def_(min, minmax_from_sets(3, integer_set|float_set))

builtins += def_(max, minmax_from_sets(2, integer_set|float_set))
builtins += def_(max, minmax_from_sets(3, integer_set|float_set))

def get_builtin_function_library(lib=None):
    '''Create or add builtin functions to a FunctionLibrary instance.
    '''
    lib = FunctionLibrary() if lib is None else lib
    for k, vl in builtins:
        for args, return_type in vl:
            fn = Function(funcobj=k, args=args, return_type=return_type)            
            lib.define(fn)
    return lib

