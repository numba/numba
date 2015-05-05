from __future__ import absolute_import, print_function

import numpy
from .. import types
from .templates import (AttributeTemplate, AbstractTemplate,
                        AbstractKeywordTemplate, Registry, signature)

from ..numpy_support import (ufunc_find_matching_loop,
                             supported_ufunc_loop, as_dtype,
                             from_dtype)

from ..typeinfer import TypingError

registry = Registry()
builtin = registry.register
builtin_global = registry.register_global
builtin_attr = registry.register_attr


@builtin_attr
class NumpyModuleAttribute(AttributeTemplate):
    # note: many unary ufuncs are added later on, using setattr
    key = types.Module(numpy)


class Numpy_rules_ufunc(AbstractTemplate):
    @classmethod
    def _handle_inputs(cls, ufunc, args, kws):
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
        return base_types, explicit_outputs, ndims

    def generic(self, args, kws):
        ufunc = self.key
        base_types, explicit_outputs, ndims = self._handle_inputs(ufunc, args,
                                                                  kws)
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
        implicit_output_count = ufunc.nout - len(explicit_outputs)
        if implicit_output_count > 0:
            # XXX this is currently wrong for datetime64 and timedelta64,
            # as ufunc_find_matching_loop() doesn't do any type inference.
            ret_tys = ufunc_loop.outputs[-implicit_output_count:]
            if ndims > 0:
                # XXX Not sure 'A' layout is correct...
                ret_tys = [types.Array(dtype=ret_ty, ndim=ndims, layout='A')
                           for ret_ty in ret_tys]
            out.extend(ret_tys)

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

all_ufuncs = sum([_math_operations, _trigonometric_functions,
                  _bit_twiddling_functions, _comparison_functions,
                  _floating_functions], [])

supported_ufuncs = [x for x in all_ufuncs if x not in _unsupported]

for func in supported_ufuncs:
    _numpy_ufunc(func)

all_ufuncs = [getattr(numpy, name) for name in all_ufuncs]
supported_ufuncs = [getattr(numpy, name) for name in supported_ufuncs]


del _math_operations, _trigonometric_functions, _bit_twiddling_functions
del _comparison_functions, _floating_functions, _unsupported
del _aliases, _numpy_ufunc


# -----------------------------------------------------------------------------
# Install global reduction functions

# Functions where input domain and output domain are the same
class Numpy_homogenous_reduction(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        [arr] = args
        return signature(arr.dtype, arr)

# Functions where domain and range are possibly different formats
class Numpy_expanded_reduction(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        [arr] = args
        if isinstance(arr.dtype, types.Integer):
            # Expand to a machine int, not larger (like Numpy)
            if arr.dtype.signed:
                return signature(max(arr.dtype, types.intp), arr)
            else:
                return signature(max(arr.dtype, types.uintp), arr)
        else:
            return signature(arr.dtype, arr)

class Numpy_heterogenous_reduction_real(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        [arr] = args
        if arr.dtype in types.integer_domain:
            return signature(types.float64, arr)
        else:
            return signature(arr.dtype, arr)

class Numpy_index_reduction(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        [arr] = args
        return signature(types.int64, arr)

# Function to glue attributes onto the numpy-esque object
def _numpy_reduction(fname, rClass):
    npyfn = getattr(numpy, fname)
    cls = type("Numpy_reduce_{0}".format(npyfn), (rClass,), dict(key=npyfn))
    semiBound = lambda self, mod: types.Function(cls)
    setattr(NumpyModuleAttribute, "resolve_{0}".format(fname), semiBound)

for func in ['min', 'max']:
    _numpy_reduction(func, Numpy_homogenous_reduction)

for func in ['sum', 'prod']:
    _numpy_reduction(func, Numpy_expanded_reduction)

for func in ['mean', 'var', 'std']:
    _numpy_reduction(func, Numpy_heterogenous_reduction_real)

for func in ['argmin', 'argmax']:
    _numpy_reduction(func, Numpy_index_reduction)


# -----------------------------------------------------------------------------
# Numpy scalar constructors

# Register numpy.int8, etc. as convertors to the equivalent Numba types
np_types = set(getattr(numpy, str(nb_type)) for nb_type in types.number_domain)
# Those may or may not be aliases (depending on the Numpy build / version)
np_types.add(numpy.intc)
np_types.add(numpy.intp)
np_types.add(numpy.uintc)
np_types.add(numpy.uintp)


def register_casters(register_global):
    for np_type in np_types:
        nb_type = getattr(types, np_type.__name__)

        class Caster(AbstractTemplate):
            key = np_type
            restype = nb_type

            def generic(self, args, kws):
                assert not kws
                [a] = args
                if a in types.number_domain:
                    return signature(self.restype, a)

        register_global(np_type, types.Function(Caster))

register_casters(builtin_global)

# -----------------------------------------------------------------------------
# Miscellaneous functions

@builtin
class NdEnumerate(AbstractTemplate):
    key = numpy.ndenumerate

    def generic(self, args, kws):
        assert not kws
        arr, = args

        if isinstance(arr, types.Array):
            enumerate_type = types.NumpyNdEnumerateType(arr)
            return signature(enumerate_type, *args)

builtin_global(numpy.ndenumerate, types.Function(NdEnumerate))

@builtin
class NdIndex(AbstractTemplate):
    key = numpy.ndindex

    def generic(self, args, kws):
        assert not kws

        # Either ndindex(shape) or ndindex(*shape)
        if len(args) == 1 and isinstance(args[0], types.UniTuple):
            shape = list(args[0])
        else:
            shape = args

        if shape and all(isinstance(x, types.Integer) for x in shape):
            iterator_type = types.NumpyNdIndexType(len(shape))
            return signature(iterator_type, *args)

builtin_global(numpy.ndindex, types.Function(NdIndex))


@builtin
class NdEmpty(AbstractKeywordTemplate):
    key = numpy.empty
    keywords = 'shape', 'dtype'

    def generic_keywords(self, kwargs):
        shape = kwargs['shape']
        dtype = kwargs.get('dtype')
        np_dtype = types.double
        if dtype is not None:
            # numpy APIs allow dtype constructor to be used as `dtype`
            # arguments.  Since, npy_dtype.template.key dtype or dtype
            # ctor, we use numpy.dtype to force it into a dtype object.
            np_dtype = from_dtype(numpy.dtype(dtype.template.key))

        if isinstance(shape, types.Integer):
            ndim = 1
        elif isinstance(shape, types.BaseTuple):
            ndim = len(shape)
            if not all(isinstance(s, types.Integer) for s in shape):
                # Not all element in shape are integer
                return
        else:
            return

        return_type = types.Array(dtype=np_dtype, ndim=ndim, layout='C')

        args = [shape]
        keywords = set(['shape'])
        if dtype is not None:
            args.append(dtype)
            keywords.add('dtype')

        return signature(return_type, *args, keywords=keywords)


builtin_global(numpy.empty, types.Function(NdEmpty))

@builtin
class Round(AbstractTemplate):
    key = numpy.round

    def generic(self, args, kws):
        assert not kws
        assert 1 <= len(args) <= 3

        arg = args[0]
        if len(args) == 1:
            decimals = types.int32
            out = None
        else:
            decimals = args[1]
            if len(args) == 2:
                out = None
            else:
                out = args[2]

        supported_scalars = (types.Integer, types.Float, types.Complex)
        if isinstance(arg, supported_scalars):
            assert out is None
            return signature(arg, *args)
        if (isinstance(arg, types.Array) and isinstance(arg.dtype, supported_scalars) and
            isinstance(out, types.Array) and isinstance(out.dtype, supported_scalars) and
            out.ndim == arg.ndim):
            # arg can only be complex if out is complex too
            if (not isinstance(arg.dtype, types.Complex)
                or isinstance(out.dtype, types.Complex)):
                return signature(out, *args)

builtin_global(numpy.round, types.Function(Round))
builtin_global(numpy.around, types.Function(Round))


builtin_global(numpy, types.Module(numpy))

