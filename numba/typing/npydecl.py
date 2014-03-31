from __future__ import absolute_import, print_function

import numpy
from .. import types
from .templates import (AttributeTemplate, AbstractTemplate,
                                    Registry, signature)

registry = Registry()
builtin_global = registry.register_global
builtin_attr = registry.register_attr


_typemap = {
    '?': types.bool_,
    'b': types.int8,
    'B': types.uint8,
    'h': types.short,
    'H': types.ushort,
    'i': types.int32, # should be C int
    'I': types.uint32, # should be C unsigned int
    'l': types.long_,
    'L': types.ulong,
    'q': types.longlong,
    'Q': types.ulonglong,

    'f': types.float_,
    'd': types.double,
#    'g': types.longdouble,
    'F': types.complex64,  # cfloat
    'D': types.complex128, # cdouble
#   'G': types.clongdouble
    'O': types.pyobject,
    'M': types.pyobject
}

_inv_typemap = { v: k  for k,v in _typemap.iteritems() }

@builtin_attr
class NumpyModuleAttribute(AttributeTemplate):
    # note: many unary ufuncs are added later on, using setattr
    key = types.Module(numpy)

    def resolve_arctan2(self, mod):
        return types.Function(Numpy_arctan2)

    def resolve_add(self, mod):
        return types.Function(Numpy_add)

    def resolve_subtract(self, mod):
        return types.Function(Numpy_subtract)

    def resolve_multiply(self, mod):
        return types.Function(Numpy_multiply)

    def resolve_divide(self, mod):
        return types.Function(Numpy_divide)


class Numpy_rules_ufunc(AbstractTemplate):
    def generic(self, args, kws):
        assert(self.key.nout == 1) # this function assumes only one output

        if len(args) > self.key.nin:
            # more args than inputs... assume everything is typed :)
            assert(len(args) == self.key.nargs)
            
            return signature(args[-1], *args)

        # else... we must look for the kernel to use, the actual loop that
        # will be used, using NumPy's logic:
        assert(len(args) == self.key.nin)
        letter_arg_types = [ _inv_typemap[x.dtype if isinstance(x, types.Array) else x] for x in args ]

        for candidate in self.key.types:
            if numpy.alltrue([numpy.can_cast(*x) 
                              for x in zip(letter_arg_types,
                                           candidate[0:self.key.nin])]):
                #found!
                array_arg = [isinstance(a, types.Array) for a in args]
                out = _typemap[candidate[-1]]
                if any(array_arg):
                    ndims = max(*[a.ndim if isinstance(a, types.Array) else 0 for a in args])
                    out = types.Array(out, ndims, 'A')
                return signature(out, *args)

        # At this point if we don't have a candidate, we are out of luck. NumPy won't know
        # how to eval this!
        raise TypingError("can't resolve ufunc {0} for types {1}".format(key.__name__, args))


def _numpy_ufunc(name):
    the_key = eval("numpy."+name) # obtain the appropriate symbol for the key.
    class typing_class(Numpy_rules_ufunc):
        key = the_key

    # Add the resolve method to NumpyModuleAttribute
    setattr(NumpyModuleAttribute, "resolve_"+name, lambda s, m: types.Function(typing_class))
    builtin_global(the_key, types.Function(typing_class))


# list of unary ufuncs to register
_autoregister_ufuncs = [
    "sin", "cos", "tan", "arcsin", "arccos", "arctan",
    "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh",
    "exp", "exp2", "expm1",
    "log", "log2", "log10", "log1p",
    "absolute", "negative", "floor", "ceil", "trunc", "sign",
    "sqrt",
    "deg2rad", "rad2deg",
    "add", "subtract", "multiply", "divide",
    "arctan2"]
for func in _autoregister_ufuncs:
    _numpy_ufunc(func)
del(_autoregister_ufuncs)


builtin_global(numpy, types.Module(numpy))
