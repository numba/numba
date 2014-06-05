from __future__ import absolute_import, print_function

import numpy
from .. import types
from .templates import (AttributeTemplate, AbstractTemplate,
                                    Registry, signature)

from ..numpy_support import (ufunc_find_matching_loop,
                             numba_types_to_numpy_letter_types,
                             numpy_letter_types_to_numba_types)

registry = Registry()
builtin_global = registry.register_global
builtin_attr = registry.register_attr

@builtin_attr
class NumpyModuleAttribute(AttributeTemplate):
    # note: many unary ufuncs are added later on, using setattr
    key = types.Module(numpy)


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
        base_types = [x.dtype if isinstance(x, types.Array) else x for x in args]
        letter_arg_types = numba_types_to_numpy_letter_types(base_types)

        ufunc_loop = ufunc_find_matching_loop(self.key, letter_arg_types)
        if ufunc_loop is not None:
            # a result was found so...
            array_arg = [isinstance(a, types.Array) for a in args]
            # base out type will be based on the ufunc result type (last letter)
            out = numpy_letter_types_to_numba_types(ufunc_loop[-self.key.nout:])
            if any(array_arg):
                # if any argument was an array, the result will be an array
                ndims = max(*[a.ndim if isinstance(a, types.Array) else 0 for a in args])
                out = [types.Array(x, ndims, 'A') for x in out] 
            out.extend(args)
            return signature(*out)

        # At this point if we don't have a candidate, we are out of luck. NumPy won't know
        # how to eval this!
        raise TypingError("can't resolve ufunc {0} for types {1}".format(key.__name__, args))


def _numpy_ufunc(name):
    the_key = eval("numpy."+name) # obtain the appropriate symbol for the key.
    class typing_class(Numpy_rules_ufunc):
        key = the_key

    typing_class.__name__ = "resolve_{0}".format(name)
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
    "arctan2", "power"]
for func in _autoregister_ufuncs:
    _numpy_ufunc(func)
del(_autoregister_ufuncs)


builtin_global(numpy, types.Module(numpy))
