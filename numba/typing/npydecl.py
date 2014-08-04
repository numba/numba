from __future__ import absolute_import, print_function

import numpy
import itertools
from .. import types
from .templates import (AttributeTemplate, AbstractTemplate,
                                    Registry, signature)

from ..numpy_support import (ufunc_find_matching_loop,
                             numba_types_to_numpy_letter_types,
                             numpy_letter_types_to_numba_types,
                             supported_letter_types)

from ..typeinfer import TypingError

registry = Registry()
builtin_global = registry.register_global
builtin_attr = registry.register_attr

@builtin_attr
class NumpyModuleAttribute(AttributeTemplate):
    # note: many unary ufuncs are added later on, using setattr
    key = types.Module(numpy)


class Numpy_rules_ufunc(AbstractTemplate):
    def generic(self, args, kws):
        ufunc = self.key
        nin = ufunc.nin
        nout = ufunc.nout
        nargs = ufunc.nargs

        if nout > 1:
            msg = "ufunc {0} not supported in this mode (more than 1 output)"
            raise TypingError(msg=msg.format(ufunc.__name__))

        # preconditions
        assert nargs == nin + nout
        assert len(args) >= nin and len(args) <= nargs

        arg_ndims = [a.ndim if isinstance(a, types.Array) else 0 for a in args]
        ndims = arg_ndims[0] if len(arg_ndims) < 2 else max(*arg_ndims)

        # explicit outputs must be arrays (no explicit scalar return values supported)
        explicit_outputs = args[nin:]

        # all the explicit outputs must match the number max number of dimensions
        if not all((d == ndims for d in arg_ndims[nin:])):
            msg = "ufunc '{0}' called with unsuitable explicit output arrays."
            raise TypingError(msg=msg.format(ufunc.__name__))

        if not all((isinstance(output, types.Array) for output in explicit_outputs)):
            msg = "ufunc '{0}' called with an explicit output that is not an array"
            raise TypingError(msg=msg.format(ufunc.__name__))
        
        # find the kernel to use, based only in the input types (as does NumPy)
        base_types = [x.dtype if isinstance(x, types.Array) else x for x in args]
        letter_arg_types = numba_types_to_numpy_letter_types(base_types[:nin])
        ufunc_loop = ufunc_find_matching_loop(ufunc, letter_arg_types)
        if ufunc_loop is None:
            TypingError("can't resolve ufunc {0} for types {1}".format(ufunc.__name__, args))

        ufunc_loop_types = ufunc_loop[:ufunc.nin] + ufunc_loop[-ufunc.nout:]
        supported_types = supported_letter_types()
        # check if all the types involved in the ufunc loop are supported in this mode
        if any((t not in supported_types for t in ufunc_loop_types)):
            msg = "ufunc '{0}' using the loop '{1}' not supported in this mode"
            raise TypingError(msg=msg.format(ufunc.__name__, ufunc_loop))

        # if there is any explicit output type, check that it is valid
        explicit_outputs_np = ''.join(numba_types_to_numpy_letter_types(
            [ty.dtype for ty in explicit_outputs]))

        # Numpy will happily use unsafe conversions (although it will actually warn)
        if not all ((numpy.can_cast(fromty, toty, 'unsafe') for fromty, toty in
                     zip(ufunc_loop_types[-nout], explicit_outputs_np))):
            msg = "ufunc '{0}' can't cast result to explicit result type"
            raise TypingError(msg=msg.format(ufunc.__name__))

        # a valid loop was found that is compatible. The result of type inference should
        # be based on the explicit output types, and when not available with the type given
        # by the selected NumPy loop
        out = list(explicit_outputs)
        if nout > len(explicit_outputs):
            implicit_letter_types = ufunc_loop_types[len(explicit_outputs)-nout:]
            implicit_out_types = numpy_letter_types_to_numba_types(implicit_letter_types)
            if ndims:
                implicit_out_types = [types.Array(t, ndims, 'A') for t in implicit_out_types]

            out.extend(implicit_out_types)

        # note: although the previous code should support multiple return values, only one
        #       is supported as of now (signature may not support more than one).
        #       there is an assert enforcing only one output
        out.extend(args)
        return signature(*out)


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
