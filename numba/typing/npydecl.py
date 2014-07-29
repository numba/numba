from __future__ import absolute_import, print_function

import numpy
import itertools
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


# list of unary ufuncs to register

_math_operations = [ "add", "subtract", "multiply",
                     "logaddexp", "logaddexp2", "true_divide",
                     "floor_divide", "negative", "power", 
                     "remainder", "fmod", "absolute",
                     "rint", "sign", "conj", "exp", "exp2",
                     "log", "log2", "log10", "expm1", "log1p",
                     "sqrt", "square", "reciprocal",
                     "divide", "mod", "abs", "fabs" ]

_trigonometric_functions = [ "sin", "cos", "tan", "arcsin",
                             "arccos", "arctan", "arctan2",
                             "hypot", "sinh", "cosh", "tanh",
                             "arcsinh", "arccosh", "arctanh",
                             "deg2rad", "rad2deg", "degrees",
                             "radians" ]

_bit_twiddling_functions = ["bitwise_and", "bitwise_or",
                            "bitwise_xor", "invert", 
                            "left_shift", "right_shift",
                            "bitwise_not" ]

_comparison_functions = [ "greater", "greater_equal", "less",
                          "less_equal", "not_equal", "equal",
                          "logical_and", "logical_or",
                          "logical_xor", "logical_not",
                          "maximum", "minimum", "fmax", "fmin" ]

_floating_functions = [ "isfinite", "isinf", "isnan", "signbit",
                        "copysign", "nextafter", "modf", "ldexp",
                        "frexp", "floor", "ceil", "trunc",
                        "spacing" ]


# This is a set of the ufuncs that are not yet supported by Lowering. In order
# to trigger no-python mode we must not register them until their Lowering is
# implemented.
#
# It also works as a nice TODO list for ufunc support :)
_unsupported = set([ numpy.square, numpy.spacing, numpy.signbit,
                     numpy.right_shift, numpy.remainder, numpy.reciprocal,
                     numpy.not_equal, numpy.minimum, numpy.maximum,
                     numpy.logical_xor, numpy.logical_or, numpy.logical_not,
                     numpy.logical_and, numpy.less,
                     numpy.less_equal, numpy.left_shift, numpy.isnan, numpy.isinf,
                     numpy.isfinite, numpy.invert, numpy.greater,
                     numpy.greater_equal, numpy.fmod, numpy.fmin, numpy.fmax,
                     numpy.equal, numpy.copysign,
                     numpy.conjugate, numpy.bitwise_xor,
                     numpy.bitwise_or, numpy.bitwise_and ])

# a list of ufuncs that are in fact aliases of other ufuncs. They need to insert the
# resolve method, but not register the ufunc itself
_aliases = set(["bitwise_not", "mod", "abs"])

#in python3 numpy.divide is mapped to numpy.true_divide
if numpy.divide == numpy.true_divide:
    _aliases.add("divide")



def _numpy_ufunc(name):
    the_key = eval("numpy."+name) # obtain the appropriate symbol for the key.
    class typing_class(Numpy_rules_ufunc):
        key = the_key

    typing_class.__name__ = "resolve_{0}".format(name)
    # Add the resolve method to NumpyModuleAttribute
    setattr(NumpyModuleAttribute, "resolve_"+name, lambda s, m: types.Function(typing_class))

    if not name in _aliases:
        builtin_global(the_key, types.Function(typing_class))


for func in itertools.chain(_math_operations, _trigonometric_functions,
                            _bit_twiddling_functions, _comparison_functions,
                            _floating_functions):
    if not getattr(numpy, func) in _unsupported:
        _numpy_ufunc(func)


del _math_operations, _trigonometric_functions, _bit_twiddling_functions
del _comparison_functions, _floating_functions, _unsupported
del _aliases, _numpy_ufunc

builtin_global(numpy, types.Module(numpy))
