from __future__ import absolute_import, print_function

import numpy
from .. import types, utils
from .templates import (AttributeTemplate, AbstractTemplate, CallableTemplate,
                        Registry, signature)

from ..numpy_support import (ufunc_find_matching_loop,
                             supported_ufunc_loop, as_dtype,
                             from_dtype)
from ..numpy_support import version as numpy_version

from ..errors import TypingError

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

            # Prefer C contig if any array is C contig.
            # Next, prefer F contig.
            # Defaults to C contig if not layouts are C/F.
            if 'C' not in layouts and 'F' in layouts:
                layout = 'F'

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

    def generic(self, args, kws):
        '''Overloads and calls base class generic() method, returning
        None if a TypingError occurred.

        Returning None for operators is important since operators are
        heavily overloaded, and by suppressing type errors, we allow
        type inference to check other possibilities before giving up
        (particularly user-defined operators).
        '''
        try:
            sig = super(NumpyRulesArrayOperator, self).generic(args, kws)
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


_binop_map = NumpyRulesArrayOperator._op_map

class NumpyRulesInplaceArrayOperator(NumpyRulesArrayOperator):
    _op_map = dict((inp, _binop_map[binop])
                   for (inp, binop) in utils.inplace_map.items()
                   if binop in _binop_map)

    def generic(self, args, kws):
        # Type the inplace operator as if an explicit output was passed,
        # to handle type resolution correctly.
        # (for example int8[:] += int16[:] should use an int8[:] output,
        #  not int16[:])
        lhs, rhs = args
        if not isinstance(lhs, types.Array):
            return
        args = args + (lhs,)
        sig = super(NumpyRulesInplaceArrayOperator, self).generic(args, kws)
        # Strip off the fake explicit output
        assert len(sig.args) == 3
        real_sig = signature(sig.return_type, *sig.args[:2])
        return real_sig


class NumpyRulesUnaryArrayOperator(NumpyRulesArrayOperator):
    _op_map = {
        # Positive is a special case since there is no Numpy ufunc
        # corresponding to it (it's essentially an identity operator).
        # See UnaryPositiveArray, above.
        '-': "negative",
        '~': "invert",
    }

    def generic(self, args, kws):
        assert not kws
        if len(args) == 1 and isinstance(args[0], types.Array):
            return super(NumpyRulesUnaryArrayOperator, self).generic(args, kws)


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
NumpyRulesInplaceArrayOperator.install_operations()

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
    cls = type("Numpy_redirect_{0}".format(fname), (Numpy_method_redirection,),
               dict(key=numpy_function, method_name=fname))
    builtin_global(numpy_function, types.Function(cls))

for func in ['min', 'max', 'sum', 'prod', 'mean', 'median', 'var', 'std',
             'cumsum', 'cumprod', 'argmin', 'argmax', 'nonzero']:
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


def register_number_classes(register_global):
    for np_type in np_types:
        nb_type = getattr(types, np_type.__name__)

        register_global(np_type, types.NumberClass(nb_type))


register_number_classes(builtin_global)


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
        """
        np.empty_like(array) -> empty array of the same shape and layout
        np.empty_like(scalar) -> empty 0-d array of the scalar type
        """
        def typer(arg, dtype=None):
            if dtype is not None:
                nb_dtype = _parse_dtype(dtype)
            elif isinstance(arg, types.Array):
                nb_dtype = arg.dtype
            else:
                nb_dtype = arg
            if nb_dtype is not None:
                if isinstance(arg, types.Array):
                    layout = arg.layout if arg.layout != 'A' else 'C'
                    return arg.copy(dtype=nb_dtype, layout=layout)
                else:
                    return types.Array(nb_dtype, 0, 'C')

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
            """
            np.full_like(array, val) -> array of the same shape and layout
            np.full_like(scalar, val) -> 0-d array of the scalar type
            """
            def typer(arg, fill_value, dtype=None):
                if dtype is not None:
                    nb_dtype = _parse_dtype(dtype)
                elif isinstance(arg, types.Array):
                    nb_dtype = arg.dtype
                else:
                    nb_dtype = arg
                if nb_dtype is not None:
                    if isinstance(arg, types.Array):
                        return arg.copy(dtype=nb_dtype)
                    else:
                        return types.Array(dtype=nb_dtype, ndim=0, layout='C')

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


@builtin
class NdFromBuffer(CallableTemplate):
    key = numpy.frombuffer

    def generic(self):
        def typer(buffer, dtype=None):
            if not isinstance(buffer, types.Buffer) or buffer.layout != 'C':
                return
            if dtype is None:
                nb_dtype = types.float64
            else:
                nb_dtype = _parse_dtype(dtype)

            if nb_dtype is not None:
                return types.Array(dtype=nb_dtype, ndim=1, layout='C',
                                   readonly=not buffer.mutable)

        return typer

builtin_global(numpy.frombuffer, types.Function(NdFromBuffer))


@builtin
class NdSort(CallableTemplate):
    key = numpy.sort

    def generic(self):
        def typer(a):
            if isinstance(a, types.Array) and a.ndim == 1:
                return a

        return typer

builtin_global(numpy.sort, types.Function(NdSort))


# -----------------------------------------------------------------------------
# Linear algebra


class MatMulTyperMixin(object):

    def matmul_typer(self, a, b, out=None):
        """
        Typer function for Numpy matrix multiplication.
        """
        if not isinstance(a, types.Array) or not isinstance(b, types.Array):
            return
        if not all(x.ndim in (1, 2) for x in (a, b)):
            raise TypingError("%s only supported on 1-D and 2-D arrays"
                              % (self.func_name, ))
        # Output dimensionality
        ndims = set([a.ndim, b.ndim])
        if ndims == set([2]):
            # M * M
            out_ndim = 2
        elif ndims == set([1, 2]):
            # M* V and V * M
            out_ndim = 1
        elif ndims == set([1]):
            # V * V
            out_ndim = 0

        if out is not None:
            if out_ndim == 0:
                raise TypeError("explicit output unsupported for vector * vector")
            elif out.ndim != out_ndim:
                raise TypeError("explicit output has incorrect dimensionality")
            if not isinstance(out, types.Array) or out.layout != 'C':
                raise TypeError("output must be a C-contiguous array")
            all_args = (a, b, out)
        else:
            all_args = (a, b)

        if not all(x.layout in 'CF' for x in (a, b)):
            # Numpy seems to allow non-contiguous arguments, but we
            # don't (we would need to copy before calling BLAS)
            raise TypingError("%s only supported on "
                              "contiguous arrays" % (self.func_name,))
        if not all(x.dtype == a.dtype for x in all_args):
            raise TypingError("%s arguments must all have "
                              "the same dtype" % (self.func_name,))
        if not isinstance(a.dtype, (types.Float, types.Complex)):
            raise TypingError("%s only supported on "
                              "float and complex arrays"
                              % (self.func_name,))
        if out:
            return out
        elif out_ndim > 0:
            return types.Array(a.dtype, out_ndim, 'C')
        else:
            return a.dtype


@builtin
class Dot(MatMulTyperMixin, CallableTemplate):
    key = numpy.dot
    func_name = "np.dot()"

    def generic(self):
        def typer(a, b, out=None):
            # NOTE: np.dot() and the '@' operator have distinct semantics
            # for >2-D arrays, but we don't support them.
            return self.matmul_typer(a, b, out)

        return typer

builtin_global(numpy.dot, types.Function(Dot))


@builtin
class VDot(CallableTemplate):
    key = numpy.vdot

    def generic(self):
        def typer(a, b):
            if not isinstance(a, types.Array) or not isinstance(b, types.Array):
                return
            if not all(x.ndim == 1 for x in (a, b)):
                raise TypingError("np.vdot() only supported on 1-D arrays")
            if not all(x.layout in 'CF' for x in (a, b)):
                # Numpy seems to allow non-contiguous arguments, but we
                # don't (we would need to copy before calling BLAS)
                raise TypingError("np.vdot() only supported on "
                                  "contiguous arrays")
            if not all(x.dtype == a.dtype for x in (a, b)):
                raise TypingError("np.vdot() arguments must all have "
                                  "the same dtype")
            if not isinstance(a.dtype, (types.Float, types.Complex)):
                raise TypingError("np.vdot() only supported on "
                                  "float and complex arrays")
            return a.dtype

        return typer

builtin_global(numpy.vdot, types.Function(VDot))


@builtin
class MatMul(MatMulTyperMixin, AbstractTemplate):
    key = "@"
    func_name = "'@'"

    def generic(self, args, kws):
        assert not kws
        restype = self.matmul_typer(*args)
        if restype is not None:
            return signature(restype, *args)


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
        if len(args) == 1 and isinstance(args[0], types.BaseTuple):
            tup = args[0]
            if tup.count > 0 and not isinstance(tup, types.UniTuple):
                # Heterogenous tuple
                return
            shape = list(tup)
        else:
            shape = args

        if all(isinstance(x, types.Integer) for x in shape):
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
            decimals = types.intp
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


@builtin
class Where(AbstractTemplate):
    key = numpy.where

    def generic(self, args, kws):
        assert not kws

        if len(args) == 1:
            # 0-dim arrays return one result array
            ary = args[0]
            ndim = max(ary.ndim, 1)
            retty = types.UniTuple(types.Array(types.intp, 1, 'C'), ndim)
            return signature(retty, ary)

        elif len(args) == 3:
            # NOTE: contrary to Numpy, we only support homogenous arguments
            cond, x, y = args
            if isinstance(cond, types.Array):
                # array where()
                if (cond.ndim == x.ndim == y.ndim and
                    x.dtype == y.dtype):
                    retty = types.Array(x.dtype, x.ndim, x.layout)
                    return signature(retty, *args)
            else:
                # scalar where()
                if not isinstance(x, types.Array) and x == y:
                    retty = types.Array(x, 0, 'C')
                    return signature(retty, *args)

builtin_global(numpy.where, types.Function(Where))


@builtin
class Sinc(AbstractTemplate):
    key = numpy.sinc

    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        arg = args[0]
        supported_scalars = (types.Float, types.Complex)
        if (isinstance(arg, supported_scalars) or
              (isinstance(arg, types.Array) and
               isinstance(arg.dtype, supported_scalars))):
            return signature(arg, arg)

builtin_global(numpy.sinc, types.Function(Sinc))


class AngleCtor(CallableTemplate):
    """
    Typing template for np.angle()
    """
    def generic(self):
        def typer(ref, deg = False):
            if isinstance(ref, types.Array):
                return ref.copy(dtype=ref.underlying_float)
            else:
                return types.float64
        return typer

@builtin
class Angle(AngleCtor):
    key = numpy.angle


builtin_global(numpy.angle, types.Function(Angle))


builtin_global(numpy, types.Module(numpy))
