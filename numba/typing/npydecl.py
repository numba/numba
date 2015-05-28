from __future__ import absolute_import, print_function

import numpy
from .. import types
from .templates import (AttributeTemplate, AbstractTemplate, CallableTemplate,
                        Registry, signature)

from ..numpy_support import (ufunc_find_matching_loop,
                             supported_ufunc_loop, as_dtype,
                             from_dtype)
from ..numpy_support import version as numpy_version

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
        base_types = [x.dtype if isinstance(x, types.Array) else x
                      for x in args]

        # Figure out the output array layout, if needed.
        layout = None
        if ndims > 0 and (len(explicit_outputs) < ufunc.nout):
            layout = 'C'
            layouts = [x.layout if isinstance(x, types.Array) else ''
                       for x in args]
            if 'C' not in layouts:
                if 'F' in layouts:
                    layout = 'F'
                elif 'A' in layouts:
                    # See also _empty_nd_impl() in numba.targets.arrayobj.
                    raise NotImplementedError(
                        "Don't know how to create implicit output array "
                        "with 'A' layout.")

        return base_types, explicit_outputs, ndims, layout

    @property
    def ufunc(self):
        return self.key

    def generic(self, args, kws):
        ufunc = self.ufunc
        base_types, explicit_outputs, ndims, layout = self._handle_inputs(
            ufunc, args, kws)
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
                assert layout is not None
                ret_tys = [types.Array(dtype=ret_ty, ndim=ndims, layout=layout)
                           for ret_ty in ret_tys]
            out.extend(ret_tys)

        # note: although the previous code should support multiple return values, only one
        #       is supported as of now (signature may not support more than one).
        #       there is an check enforcing only one output
        out.extend(args)
        return signature(*out)


@builtin
class UnaryPositiveArray(AbstractTemplate):
    '''Typing template class for +(array) expressions.  This operator is
    special because there is no Numpy ufunc associated with it; we
    include typing for it here (numba.typing.npydecl) because this is
    where the remaining array operators are defined.
    '''
    key = "+"

    def generic(self, args, kws):
        assert not kws
        if len(args) == 1 and isinstance(args[0], types.Array):
            arg_ty = args[0]
            return arg_ty.copy()(arg_ty)


class NumpyRulesArrayOperator(Numpy_rules_ufunc):
    _op_map = {
         '+': "add",
         '-': "subtract",
         '*': "multiply",
        '/?': "divide",
         '/': "true_divide",
        '//': "floor_divide",
         '%': "remainder",
        '**': "power",
        '<<': "left_shift",
        '>>': "right_shift",
         '&': "bitwise_and",
         '|': "bitwise_or",
         '^': "bitwise_xor",
        '==': "equal",
         '>': "greater",
        '>=': "greater_equal",
         '<': "less",
        '<=': "less_equal",
        '!=': "not_equal",
    }

    @property
    def ufunc(self):
        return getattr(numpy, self._op_map[self.key])

    @classmethod
    def install_operations(cls):
        for op, ufunc_name in cls._op_map.items():
            builtin(type("NumpyRulesArrayOperator_" + ufunc_name, (cls,),
                         dict(key=op)))

    def generic(self, *args, **kws):
        '''Overloads and calls base class generic() method, returning
        None if a TypingError occurred.

        Returning None for operators is important since operators are
        heavily overloaded, and by suppressing type errors, we allow
        type inference to check other possibilities before giving up
        (particularly user-defined operators).
        '''
        try:
            sig = super(NumpyRulesArrayOperator, self).generic(
                *args, **kws)
            # Stay out of the timedelta64 range and domain; already
            # handled elsewhere.
            if sig is not None:
                timedelta_test = (
                    isinstance(sig.return_type, types.NPTimedelta) or
                    (all(isinstance(argty, types.NPTimedelta)
                         for argty in sig.args)))
                if timedelta_test:
                    sig = None
        except TypingError:
            sig = None
        return sig


class NumpyRulesUnaryArrayOperator(NumpyRulesArrayOperator):
    _op_map = {
        # Positive is a special case since there is no Numpy ufunc
        # corresponding to it (it's essentially an identity operator).
        # See UnaryPositiveArray, above.
        '-': "negative",
        '~': "invert",
    }


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
_unsupported = set([ 'frexp', # this one is tricky, as it has 2 returns
                     'modf',  # this one also has 2 returns
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

NumpyRulesUnaryArrayOperator.install_operations()
NumpyRulesArrayOperator.install_operations()

supported_array_operators = set(
    NumpyRulesUnaryArrayOperator._op_map.keys()).union(
        NumpyRulesArrayOperator._op_map.keys())

del _math_operations, _trigonometric_functions, _bit_twiddling_functions
del _comparison_functions, _floating_functions, _unsupported
del _aliases, _numpy_ufunc


# -----------------------------------------------------------------------------
# Install global helpers for array methods.

class Numpy_method_redirection(AbstractTemplate):
    """
    A template redirecting a Numpy global function (e.g. np.sum) to an
    array method of the same name (e.g. ndarray.sum).
    """

    def generic(self, args, kws):
        assert not kws
        [arr] = args
        # This will return a BoundFunction
        meth_ty = self.context.resolve_getattr(arr, self.method_name)
        # Resolve arguments on the bound function
        meth_sig = self.context.resolve_function_type(meth_ty, args[1:], kws)
        if meth_sig is not None:
            return signature(meth_sig.return_type, meth_sig.recvr, *meth_sig.args)


# Function to glue attributes onto the numpy-esque object
def _numpy_redirect(fname):
    numpy_function = getattr(numpy, fname)
    cls = type("Numpy_reduce_{0}".format(fname), (Numpy_method_redirection,),
               dict(key=numpy_function, method_name=fname))
    builtin_global(numpy_function, types.Function(cls))

for func in ['min', 'max', 'sum', 'prod', 'mean', 'var', 'std',
             'cumsum', 'cumprod', 'argmin', 'argmax']:
    _numpy_redirect(func)


# -----------------------------------------------------------------------------
# Numpy scalar constructors

# Register numpy.int8, etc. as convertors to the equivalent Numba types
np_types = set(getattr(numpy, str(nb_type)) for nb_type in types.number_domain)
np_types.add(numpy.bool_)
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

        register_global(np_type, types.NumberClass(nb_type, Caster))

register_casters(builtin_global)


# -----------------------------------------------------------------------------
# Numpy array constructors

def _parse_shape(shape):
    ndim = None
    if isinstance(shape, types.Integer):
        ndim = 1
    elif isinstance(shape, (types.Tuple, types.UniTuple)):
        if all(isinstance(s, types.Integer) for s in shape):
            ndim = len(shape)
    return ndim

def _parse_dtype(dtype):
    if isinstance(dtype, types.DTypeSpec):
        return dtype.dtype


class NdConstructor(CallableTemplate):
    """
    Typing template for np.empty(), .zeros(), .ones().
    """

    def generic(self):
        def typer(shape, dtype=None):
            if dtype is None:
                nb_dtype = types.double
            else:
                nb_dtype = _parse_dtype(dtype)

            ndim = _parse_shape(shape)
            if nb_dtype is not None and ndim is not None:
                return types.Array(dtype=nb_dtype, ndim=ndim, layout='C')

        return typer


class NdConstructorLike(CallableTemplate):
    """
    Typing template for np.empty_like(), .zeros_like(), .ones_like().
    """

    def generic(self):
        def typer(arr, dtype=None):
            if dtype is None:
                nb_dtype = arr.dtype
            else:
                nb_dtype = _parse_dtype(dtype)
            if nb_dtype is not None:
                return arr.copy(dtype=nb_dtype)

        return typer


@builtin
class NdEmpty(NdConstructor):
    key = numpy.empty

@builtin
class NdZeros(NdConstructor):
    key = numpy.zeros

@builtin
class NdOnes(NdConstructor):
    key = numpy.ones
    return_new_reference = True

@builtin
class NdEmptyLike(NdConstructorLike):
    key = numpy.empty_like

@builtin
class NdZerosLike(NdConstructorLike):
    key = numpy.zeros_like


builtin_global(numpy.empty, types.Function(NdEmpty))
builtin_global(numpy.zeros, types.Function(NdZeros))
builtin_global(numpy.ones, types.Function(NdOnes))
builtin_global(numpy.empty_like, types.Function(NdEmptyLike))
builtin_global(numpy.zeros_like, types.Function(NdZerosLike))

if numpy_version >= (1, 7):
    # In Numpy 1.6, ones_like() was a ufunc and had a different signature.
    @builtin
    class NdOnesLike(NdConstructorLike):
        key = numpy.ones_like

    builtin_global(numpy.ones_like, types.Function(NdOnesLike))


if numpy_version >= (1, 8):
    @builtin
    class NdFull(CallableTemplate):
        key = numpy.full
        return_new_reference = True


        def generic(self):
            def typer(shape, fill_value, dtype=None):
                if dtype is None:
                    nb_dtype = fill_value
                else:
                    nb_dtype = _parse_dtype(dtype)

                ndim = _parse_shape(shape)
                if nb_dtype is not None and ndim is not None:
                    return types.Array(dtype=nb_dtype, ndim=ndim, layout='C')

            return typer

    @builtin
    class NdFullLike(CallableTemplate):
        key = numpy.full_like
        return_new_reference = True

        def generic(self):
            def typer(arr, fill_value, dtype=None):
                if dtype is None:
                    nb_dtype = arr.dtype
                else:
                    nb_dtype = _parse_dtype(dtype)
                if nb_dtype is not None:
                    return arr.copy(dtype=nb_dtype)

            return typer

    builtin_global(numpy.full, types.Function(NdFull))
    builtin_global(numpy.full_like, types.Function(NdFullLike))


@builtin
class NdIdentity(AbstractTemplate):
    key = numpy.identity
    return_new_reference = True

    def generic(self, args, kws):
        assert not kws
        n = args[0]
        if not isinstance(n, types.Integer):
            return
        if len(args) >= 2:
            nb_dtype = _parse_dtype(args[1])
        else:
            nb_dtype = types.float64

        if nb_dtype is not None:
            return_type = types.Array(ndim=2, dtype=nb_dtype, layout='C')
            return signature(return_type, *args)

builtin_global(numpy.identity, types.Function(NdIdentity))


def _infer_dtype_from_inputs(inputs):
    return dtype


@builtin
class NdEye(CallableTemplate):
    key = numpy.eye
    return_new_reference = True

    def generic(self):
        def typer(N, M=None, k=None, dtype=None):
            if dtype is None:
                nb_dtype = types.float64
            else:
                nb_dtype = _parse_dtype(dtype)
            if nb_dtype is not None:
                return types.Array(ndim=2, dtype=nb_dtype, layout='C')

        return typer

builtin_global(numpy.eye, types.Function(NdEye))


@builtin
class NdArange(AbstractTemplate):
    key = numpy.arange
    return_new_reference = True

    def generic(self, args, kws):
        assert not kws
        if len(args) >= 4:
            dtype = _parse_dtype(args[3])
            bounds = args[:3]
        else:
            bounds = args
            if any(isinstance(arg, types.Complex) for arg in bounds):
                dtype = types.complex128
            elif any(isinstance(arg, types.Float) for arg in bounds):
                dtype = types.float64
            else:
                dtype = max(bounds)
        if not all(isinstance(arg, types.Number) for arg in bounds):
            return
        return_type = types.Array(ndim=1, dtype=dtype, layout='C')
        return signature(return_type, *args)

builtin_global(numpy.arange, types.Function(NdArange))


@builtin
class NdLinspace(AbstractTemplate):
    key = numpy.linspace
    return_new_reference = True

    def generic(self, args, kws):
        assert not kws
        bounds = args[:2]
        if not all(isinstance(arg, types.Number) for arg in bounds):
            return
        if len(args) >= 3:
            num = args[2]
            if not isinstance(num, types.Integer):
                return
        if len(args) >= 4:
            # Not supporting the other arguments as it would require
            # keyword arguments for reasonable use.
            return
        if any(isinstance(arg, types.Complex) for arg in bounds):
            dtype = types.complex128
        else:
            dtype = types.float64
        return_type = types.Array(ndim=1, dtype=dtype, layout='C')
        return signature(return_type, *args)

builtin_global(numpy.linspace, types.Function(NdLinspace))


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
