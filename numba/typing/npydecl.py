from __future__ import absolute_import, print_function

import numpy
import itertools
from .. import types
from .templates import (AttributeTemplate, AbstractTemplate,
                                    Registry, signature)

from ..numpy_support import (ufunc_find_matching_loop,
                             supported_ufunc_loop, as_dtype)

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

        # preconditions
        assert nargs == nin + nout

        if nout > 1:
            msg = "ufunc '{0}': not supported in this mode (more than 1 output)"
            raise TypingError(msg=msg.format(ufunc.__name__))

        if len(args) < nin:
            msg = "ufunc '{0}': not enough arguments ({1} found, {2} required)"
            raise TypingError(msg=msg.format(ufunc.__name__, len(args), nin))

        if len(args) > nargs:
            msg = "ufunc '{0}': too many arguments ({1} found, {2} maximum)"
            raise TypingError(msg=msg.format(ufunc.__name__, len(args), nargs))


        arg_ndims = [a.ndim if isinstance(a, types.Array) else 0 for a in args]
        ndims = max(arg_ndims)

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
        ufunc_loop = ufunc_find_matching_loop(ufunc, base_types)
        if ufunc_loop is None:
            raise TypingError("can't resolve ufunc {0} for types {1}".format(ufunc.__name__, args))

        # check if all the types involved in the ufunc loop are supported in this mode
        if not supported_ufunc_loop(ufunc, ufunc_loop):
            msg = "ufunc '{0}' using the loop '{1}' not supported in this mode"
            raise TypingError(msg=msg.format(ufunc.__name__, ufunc_loop.ufunc_sig))

        # if there is any explicit output type, check that it is valid
        explicit_outputs_np = [as_dtype(tp.dtype) for tp in explicit_outputs]

        # Numpy will happily use unsafe conversions (although it will actually warn)
        if not all (numpy.can_cast(fromty, toty, 'unsafe') for (fromty, toty) in
                    zip(ufunc_loop.numpy_outputs, explicit_outputs_np)):
            msg = "ufunc '{0}' can't cast result to explicit result type"
            raise TypingError(msg=msg.format(ufunc.__name__))

        # A valid loop was found that is compatible. The result of type inference should
        # be based on the explicit output types, and when not available with the type given
        # by the selected NumPy loop
        out = list(explicit_outputs)
        implicit_output_count = nout - len(explicit_outputs)
        if implicit_output_count > 0:
            # XXX this is currently wrong for datetime64 and timedelta64,
            # as ufunc_find_matching_loop() doesn't do any type inference.
            out.extend(ufunc_loop.outputs[-implicit_output_count:])

        # note: although the previous code should support multiple return values, only one
        #       is supported as of now (signature may not support more than one).
        #       there is an check enforcing only one output
        out.extend(args)
        return signature(*out)


# list of unary ufuncs to register

_math_operations = [ "add", "subtract", "multiply",
                     "logaddexp", "logaddexp2", "true_divide",
                     "floor_divide", "negative", "power",
                     "remainder", "fmod", "absolute",
                     "rint", "sign", "conjugate", "exp", "exp2",
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
_unsupported = set([ numpy.frexp, # this one is tricky, as it has 2 returns
                     numpy.modf,  # this one also has 2 returns
                 ])

# a list of ufuncs that are in fact aliases of other ufuncs. They need to insert the
# resolve method, but not register the ufunc itself
_aliases = set(["bitwise_not", "mod", "abs"])

#in python3 numpy.divide is mapped to numpy.true_divide
if numpy.divide == numpy.true_divide:
    _aliases.add("divide")


def _numpy_ufunc(name):
    func = getattr(numpy, name)
    class typing_class(Numpy_rules_ufunc):
        key = func

    typing_class.__name__ = "resolve_{0}".format(name)
    # Add the resolve method to NumpyModuleAttribute
    setattr(NumpyModuleAttribute, typing_class.__name__,
            lambda s, m: types.Function(typing_class))

    if not name in _aliases:
        builtin_global(func, types.Function(typing_class))


class Numpy_binary_ufunc(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        nargs = len(args)
        if nargs == 3:
            [inp1, inp2, out] = args
            if isinstance(out, types.Array) and \
                    (isinstance(inp1, types.Array) or inp1 in types.number_domain) or \
                    (isinstance(inp2, types.Array) or inp2 in types.number_domain):
                return signature(out, inp1, inp2, out)
        elif nargs == 2:
            [inp1, inp2] = args
            if inp1 in types.number_domain and inp2 in types.number_domain:
                if hasattr(self, "scalar_out_type"):
                    return signature(self.scalar_out_type, inp1, inp2)
                else:
                    return signature(inp1, inp1, inp2)

for func in itertools.chain(_math_operations, _trigonometric_functions,
                            _bit_twiddling_functions, _comparison_functions,
                            _floating_functions):
    if not getattr(numpy, func) in _unsupported:
        _numpy_ufunc(func)


del _math_operations, _trigonometric_functions, _bit_twiddling_functions
del _comparison_functions, _floating_functions, _unsupported
del _aliases, _numpy_ufunc


# -----------------------------------------------------------------------------
# Install global reduction functions

class Numpy_generic_reduction(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        [arr] = args
        return signature(arr.dtype, arr)


def _numpy_reduction(fname):
    npyfn = getattr(numpy, fname)
    cls = type("Numpy_reduce_{0}".format(npyfn), (Numpy_generic_reduction,),
               dict(key=npyfn))
    setattr(NumpyModuleAttribute, "resolve_{0}".format(fname),
            lambda self, mod: types.Function(cls))

for func in ['sum', 'prod']:
    _numpy_reduction(func)

builtin_global(numpy, types.Module(numpy))


