from __future__ import absolute_import, print_function

import warnings

import numpy as np
import operator

from .. import types, utils
from .templates import (AttributeTemplate, AbstractTemplate, CallableTemplate,
                        Registry, signature)

from ..numpy_support import (ufunc_find_matching_loop,
                             supported_ufunc_loop, as_dtype,
                             from_dtype, as_dtype, resolve_output_type,
                             carray, farray)
from ..numpy_support import version as numpy_version
from ..errors import TypingError, PerformanceWarning
from numba import pndindex

registry = Registry()
infer = registry.register
infer_global = registry.register_global
infer_getattr = registry.register_attr


class Numpy_rules_ufunc(AbstractTemplate):
    @classmethod
    def _handle_inputs(cls, ufunc, args, kws):
        """
        Process argument types to a given *ufunc*.
        Returns a (base types, explicit outputs, ndims, layout) tuple where:
        - `base types` is a tuple of scalar types for each input
        - `explicit outputs` is a tuple of explicit output types (arrays)
        - `ndims` is the number of dimensions of the loop and also of
          any outputs, explicit or implicit
        - `layout` is the layout for any implicit output to be allocated
        """
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

        args = [a.as_array if isinstance(a, types.ArrayCompatible) else a
                for a in args]
        arg_ndims = [a.ndim if isinstance(a, types.ArrayCompatible) else 0
                     for a in args]
        ndims = max(arg_ndims)

        # explicit outputs must be arrays (no explicit scalar return values supported)
        explicit_outputs = args[nin:]

        # all the explicit outputs must match the number max number of dimensions
        if not all(d == ndims for d in arg_ndims[nin:]):
            msg = "ufunc '{0}' called with unsuitable explicit output arrays."
            raise TypingError(msg=msg.format(ufunc.__name__))

        if not all(isinstance(output, types.ArrayCompatible)
                   for output in explicit_outputs):
            msg = "ufunc '{0}' called with an explicit output that is not an array"
            raise TypingError(msg=msg.format(ufunc.__name__))

        # find the kernel to use, based only in the input types (as does NumPy)
        base_types = [x.dtype if isinstance(x, types.ArrayCompatible) else x
                      for x in args]

        # Figure out the output array layout, if needed.
        layout = None
        if ndims > 0 and (len(explicit_outputs) < ufunc.nout):
            layout = 'C'
            layouts = [x.layout if isinstance(x, types.ArrayCompatible) else ''
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
        if not all (np.can_cast(fromty, toty, 'unsafe') for (fromty, toty) in
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
                ret_tys = [resolve_output_type(self.context, args, ret_ty)
                           for ret_ty in ret_tys]
            out.extend(ret_tys)

        # note: although the previous code should support multiple return values, only one
        #       is supported as of now (signature may not support more than one).
        #       there is an check enforcing only one output
        out.extend(args)
        return signature(*out)


@infer_global(operator.pos)
class UnaryPositiveArray(AbstractTemplate):
    '''Typing template class for +(array) expressions.  This operator is
    special because there is no Numpy ufunc associated with it; we
    include typing for it here (numba.typing.npydecl) because this is
    where the remaining array operators are defined.
    '''
    key = operator.pos

    def generic(self, args, kws):
        assert not kws
        if len(args) == 1 and isinstance(args[0], types.ArrayCompatible):
            arg_ty = args[0]
            return arg_ty.copy()(arg_ty)


class NumpyRulesArrayOperator(Numpy_rules_ufunc):
    _op_map = {
        operator.add: "add",
        operator.sub: "subtract",
        operator.mul: "multiply",
        operator.truediv: "true_divide",
        operator.floordiv: "floor_divide",
        operator.mod: "remainder",
        operator.pow: "power",
        operator.lshift: "left_shift",
        operator.rshift: "right_shift",
        operator.and_: "bitwise_and",
        operator.or_: "bitwise_or",
        operator.xor: "bitwise_xor",
        operator.eq: "equal",
        operator.gt: "greater",
        operator.ge: "greater_equal",
        operator.lt: "less",
        operator.le: "less_equal",
        operator.ne: "not_equal",
    }

    if not utils.IS_PY3:
        _op_map[operator.div] = "divide"

    @property
    def ufunc(self):
        return getattr(np, self._op_map[self.key])

    @classmethod
    def install_operations(cls):
        for op, ufunc_name in cls._op_map.items():
            infer_global(op)(
                type("NumpyRulesArrayOperator_" + ufunc_name, (cls,), dict(key=op))
            )

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
        except TypingError:
            return None
        if sig is None:
            return None
        args = sig.args
        # Only accept at least one array argument, otherwise the operator
        # doesn't involve Numpy's ufunc machinery.
        if not any(isinstance(arg, types.ArrayCompatible)
                   for arg in args):
            return None
        return sig


_binop_map = NumpyRulesArrayOperator._op_map

class NumpyRulesInplaceArrayOperator(NumpyRulesArrayOperator):
    _op_map = {
        operator.iadd: "add",
        operator.isub: "subtract",
        operator.imul: "multiply",
        operator.itruediv: "true_divide",
        operator.ifloordiv: "floor_divide",
        operator.imod: "remainder",
        operator.ipow: "power",
        operator.ilshift: "left_shift",
        operator.irshift: "right_shift",
        operator.iand: "bitwise_and",
        operator.ior: "bitwise_or",
        operator.ixor: "bitwise_xor",
    }

    if not utils.IS_PY3:
        _op_map[operator.idiv] = "divide"

    def generic(self, args, kws):
        # Type the inplace operator as if an explicit output was passed,
        # to handle type resolution correctly.
        # (for example int8[:] += int16[:] should use an int8[:] output,
        #  not int16[:])
        lhs, rhs = args
        if not isinstance(lhs, types.ArrayCompatible):
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
        operator.neg: "negative",
        operator.invert: "invert",
    }

    def generic(self, args, kws):
        assert not kws
        if len(args) == 1 and isinstance(args[0], types.ArrayCompatible):
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

# A list of ufuncs that are in fact aliases of other ufuncs. They need to insert the
# resolve method, but not register the ufunc itself
_aliases = set(["bitwise_not", "mod", "abs"])

# In python3 np.divide is mapped to np.true_divide
if np.divide == np.true_divide:
    _aliases.add("divide")

def _numpy_ufunc(name):
    func = getattr(np, name)
    class typing_class(Numpy_rules_ufunc):
        key = func

    typing_class.__name__ = "resolve_{0}".format(name)

    if not name in _aliases:
        infer_global(func, types.Function(typing_class))

all_ufuncs = sum([_math_operations, _trigonometric_functions,
                  _bit_twiddling_functions, _comparison_functions,
                  _floating_functions], [])

supported_ufuncs = [x for x in all_ufuncs if x not in _unsupported]

for func in supported_ufuncs:
    _numpy_ufunc(func)

all_ufuncs = [getattr(np, name) for name in all_ufuncs]
supported_ufuncs = [getattr(np, name) for name in supported_ufuncs]

NumpyRulesUnaryArrayOperator.install_operations()
NumpyRulesArrayOperator.install_operations()
NumpyRulesInplaceArrayOperator.install_operations()

supported_array_operators = set(
    NumpyRulesUnaryArrayOperator._op_map.keys()
).union(
    NumpyRulesArrayOperator._op_map.keys()
).union(
    NumpyRulesInplaceArrayOperator._op_map.keys()
)

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
        pysig = None
        if kws:
            if self.method_name == 'sum':
                def sum_stub(arr, axis):
                    pass
                pysig = utils.pysignature(sum_stub)
            elif self.method_name == 'argsort':
                def argsort_stub(arr, kind='quicksort'):
                    pass
                pysig = utils.pysignature(argsort_stub)
            else:
                fmt = "numba doesn't support kwarg for {}"
                raise TypingError(fmt.format(self.method_name))

        arr = args[0]
        # This will return a BoundFunction
        meth_ty = self.context.resolve_getattr(arr, self.method_name)
        # Resolve arguments on the bound function
        meth_sig = self.context.resolve_function_type(meth_ty, args[1:], kws)
        if meth_sig is not None:
            return meth_sig.as_function().replace(pysig=pysig)


# Function to glue attributes onto the numpy-esque object
def _numpy_redirect(fname):
    numpy_function = getattr(np, fname)
    cls = type("Numpy_redirect_{0}".format(fname), (Numpy_method_redirection,),
               dict(key=numpy_function, method_name=fname))
    infer_global(numpy_function, types.Function(cls))
    # special case literal support for 'sum'
    if fname in ['sum', 'argsort']:
        cls.support_literals = True

for func in ['min', 'max', 'sum', 'prod', 'mean', 'var', 'std',
             'cumsum', 'cumprod', 'argmin', 'argmax', 'argsort',
             'nonzero', 'ravel']:
    _numpy_redirect(func)


# -----------------------------------------------------------------------------
# Numpy scalar constructors

# Register np.int8, etc. as converters to the equivalent Numba types
np_types = set(getattr(np, str(nb_type)) for nb_type in types.number_domain)
np_types.add(np.bool_)
# Those may or may not be aliases (depending on the Numpy build / version)
np_types.add(np.intc)
np_types.add(np.intp)
np_types.add(np.uintc)
np_types.add(np.uintp)


def register_number_classes(register_global):
    for np_type in np_types:
        nb_type = getattr(types, np_type.__name__)

        register_global(np_type, types.NumberClass(nb_type))


register_number_classes(infer_global)


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

def _parse_nested_sequence(context, typ):
    """
    Parse a (possibly 0d) nested sequence type.
    A (ndim, dtype) tuple is returned.  Note the sequence may still be
    heterogeneous, as long as it converts to the given dtype.
    """
    if isinstance(typ, (types.Buffer,)):
        raise TypingError("%r not allowed in a homogeneous sequence" % typ)
    elif isinstance(typ, (types.Sequence,)):
        n, dtype = _parse_nested_sequence(context, typ.dtype)
        return n + 1, dtype
    elif isinstance(typ, (types.BaseTuple,)):
        if typ.count == 0:
            # Mimick Numpy's behaviour
            return 1, types.float64
        n, dtype = _parse_nested_sequence(context, typ[0])
        dtypes = [dtype]
        for i in range(1, typ.count):
            _n, dtype = _parse_nested_sequence(context, typ[i])
            if _n != n:
                raise TypingError("type %r does not have a regular shape"
                                  % (typ,))
            dtypes.append(dtype)
        dtype = context.unify_types(*dtypes)
        if dtype is None:
            raise TypingError("cannot convert %r to a homogeneous type" % typ)
        return n + 1, dtype
    else:
        # Scalar type => check it's valid as a Numpy array dtype
        as_dtype(typ)
        return 0, typ



@infer_global(np.array)
class NpArray(CallableTemplate):
    """
    Typing template for np.array().
    """

    def generic(self):
        def typer(object, dtype=None):
            ndim, seq_dtype = _parse_nested_sequence(self.context, object)
            if dtype is None:
                dtype = seq_dtype
            else:
                dtype = _parse_dtype(dtype)
                if dtype is None:
                    return
            return types.Array(dtype, ndim, 'C')

        return typer


@infer_global(np.empty)
@infer_global(np.zeros)
@infer_global(np.ones)
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


@infer_global(np.empty_like)
@infer_global(np.zeros_like)
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
                    return arg.copy(dtype=nb_dtype, layout=layout, readonly=False)
                else:
                    return types.Array(nb_dtype, 0, 'C')

        return typer


infer_global(np.ones_like)(NdConstructorLike)


if numpy_version >= (1, 8):
    @infer_global(np.full)
    class NdFull(CallableTemplate):

        def generic(self):
            def typer(shape, fill_value, dtype=None):
                if dtype is None:
                    if numpy_version < (1, 12):
                        nb_dtype = types.float64
                    else:
                        nb_dtype = fill_value
                else:
                    nb_dtype = _parse_dtype(dtype)

                ndim = _parse_shape(shape)
                if nb_dtype is not None and ndim is not None:
                    return types.Array(dtype=nb_dtype, ndim=ndim, layout='C')

            return typer

    @infer_global(np.full_like)
    class NdFullLike(CallableTemplate):

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
                        return arg.copy(dtype=nb_dtype, readonly=False)
                    else:
                        return types.Array(dtype=nb_dtype, ndim=0, layout='C')

            return typer


@infer_global(np.identity)
class NdIdentity(AbstractTemplate):

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


def _infer_dtype_from_inputs(inputs):
    return dtype


@infer_global(np.arange)
class NdArange(AbstractTemplate):

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


@infer_global(np.linspace)
class NdLinspace(AbstractTemplate):

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


@infer_global(np.frombuffer)
class NdFromBuffer(CallableTemplate):

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


@infer_global(np.sort)
class NdSort(CallableTemplate):

    def generic(self):
        def typer(a):
            if isinstance(a, types.Array) and a.ndim == 1:
                return a

        return typer


@infer_global(np.asfortranarray)
class AsFortranArray(CallableTemplate):

    def generic(self):
        def typer(a):
            if isinstance(a, types.Array):
                return a.copy(layout='F', ndim=max(a.ndim, 1))

        return typer


@infer_global(np.ascontiguousarray)
class AsContiguousArray(CallableTemplate):

    def generic(self):
        def typer(a):
            if isinstance(a, types.Array):
                return a.copy(layout='C', ndim=max(a.ndim, 1))

        return typer


@infer_global(np.copy)
class NdCopy(CallableTemplate):

    def generic(self):
        def typer(a):
            if isinstance(a, types.Array):
                layout = 'F' if a.layout == 'F' else 'C'
                return a.copy(layout=layout, readonly=False)

        return typer


@infer_global(np.expand_dims)
class NdExpandDims(CallableTemplate):

    def generic(self):
        def typer(a, axis):
            if (not isinstance(a, types.Array)
                or not isinstance(axis, types.Integer)):
                return

            layout = a.layout if a.ndim <= 1 else 'A'
            return a.copy(ndim=a.ndim + 1, layout=layout)

        return typer


class BaseAtLeastNdTemplate(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        if not args or not all(isinstance(a, types.Array) for a in args):
            return

        rets = [self.convert_array(a) for a in args]
        if len(rets) > 1:
            retty = types.BaseTuple.from_types(rets)
        else:
            retty = rets[0]
        return signature(retty, *args)


@infer_global(np.atleast_1d)
class NdAtLeast1d(BaseAtLeastNdTemplate):

    def convert_array(self, a):
        return a.copy(ndim=max(a.ndim, 1))


@infer_global(np.atleast_2d)
class NdAtLeast2d(BaseAtLeastNdTemplate):

    def convert_array(self, a):
        return a.copy(ndim=max(a.ndim, 2))


@infer_global(np.atleast_3d)
class NdAtLeast3d(BaseAtLeastNdTemplate):

    def convert_array(self, a):
        return a.copy(ndim=max(a.ndim, 3))


def _homogeneous_dims(context, func_name, arrays):
    ndim = arrays[0].ndim
    for a in arrays:
        if a.ndim != ndim:
            raise TypeError("%s(): all the input arrays "
                            "must have same number of dimensions"
                            % func_name)
    return ndim

def _sequence_of_arrays(context, func_name, arrays,
                        dim_chooser=_homogeneous_dims):
    if (not isinstance(arrays, types.BaseTuple)
        or not len(arrays)
        or not all(isinstance(a, types.Array) for a in arrays)):
        raise TypeError("%s(): expecting a non-empty tuple of arrays, "
                        "got %s" % (func_name, arrays))

    ndim = dim_chooser(context, func_name, arrays)

    dtype = context.unify_types(*(a.dtype for a in arrays))
    if dtype is None:
        raise TypeError("%s(): input arrays must have "
                        "compatible dtypes" % func_name)

    return dtype, ndim

def _choose_concatenation_layout(arrays):
    # Only create a F array if all input arrays have F layout.
    # This is a simplified version of Numpy's behaviour,
    # while Numpy's actually processes the input strides to
    # decide on optimal output strides
    # (see PyArray_CreateMultiSortedStridePerm()).
    return 'F' if all(a.layout == 'F' for a in arrays) else 'C'


@infer_global(np.concatenate)
class NdConcatenate(CallableTemplate):

    def generic(self):
        def typer(arrays, axis=None):
            if axis is not None and not isinstance(axis, types.Integer):
                # Note Numpy allows axis=None, but it isn't documented:
                # https://github.com/numpy/numpy/issues/7968
                return

            dtype, ndim = _sequence_of_arrays(self.context,
                                              "np.concatenate", arrays)
            if ndim == 0:
                raise TypeError("zero-dimensional arrays cannot be concatenated")

            layout = _choose_concatenation_layout(arrays)

            return types.Array(dtype, ndim, layout)

        return typer


if numpy_version >= (1, 10):
    @infer_global(np.stack)
    class NdStack(CallableTemplate):

        def generic(self):
            def typer(arrays, axis=None):
                if axis is not None and not isinstance(axis, types.Integer):
                    # Note Numpy allows axis=None, but it isn't documented:
                    # https://github.com/numpy/numpy/issues/7968
                    return

                dtype, ndim = _sequence_of_arrays(self.context,
                                                  "np.stack", arrays)

                # This diverges from Numpy's behaviour, which simply inserts
                # a new stride at the requested axis (therefore can return
                # a 'A' array).
                layout = 'F' if all(a.layout == 'F' for a in arrays) else 'C'

                return types.Array(dtype, ndim + 1, layout)

            return typer


class BaseStackTemplate(CallableTemplate):

    def generic(self):
        def typer(arrays):
            dtype, ndim = _sequence_of_arrays(self.context,
                                              self.func_name, arrays)

            ndim = max(ndim, self.ndim_min)
            layout = _choose_concatenation_layout(arrays)

            return types.Array(dtype, ndim, layout)

        return typer


@infer_global(np.hstack)
class NdStack(BaseStackTemplate):
    func_name = "np.hstack"
    ndim_min = 1

@infer_global(np.vstack)
class NdStack(BaseStackTemplate):
    func_name = "np.vstack"
    ndim_min = 2

@infer_global(np.dstack)
class NdStack(BaseStackTemplate):
    func_name = "np.dstack"
    ndim_min = 3



def _column_stack_dims(context, func_name, arrays):
    # column_stack() allows stacking 1-d and 2-d arrays together
    for a in arrays:
        if a.ndim < 1 or a.ndim > 2:
            raise TypeError("np.column_stack() is only defined on "
                            "1-d and 2-d arrays")
    return 2


@infer_global(np.column_stack)
class NdColumnStack(CallableTemplate):

    def generic(self):
        def typer(arrays):
            dtype, ndim = _sequence_of_arrays(self.context,
                                              "np.column_stack", arrays,
                                              dim_chooser=_column_stack_dims)

            layout = _choose_concatenation_layout(arrays)

            return types.Array(dtype, ndim, layout)

        return typer


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
            warnings.warn("%s is faster on contiguous arrays, called on %s"
                          % (self.func_name, (a, b)), PerformanceWarning)
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


@infer_global(np.dot)
class Dot(MatMulTyperMixin, CallableTemplate):
    func_name = "np.dot()"

    def generic(self):
        def typer(a, b, out=None):
            # NOTE: np.dot() and the '@' operator have distinct semantics
            # for >2-D arrays, but we don't support them.
            return self.matmul_typer(a, b, out)

        return typer


@infer_global(np.vdot)
class VDot(CallableTemplate):

    def generic(self):
        def typer(a, b):
            if not isinstance(a, types.Array) or not isinstance(b, types.Array):
                return
            if not all(x.ndim == 1 for x in (a, b)):
                raise TypingError("np.vdot() only supported on 1-D arrays")
            if not all(x.layout in 'CF' for x in (a, b)):
                warnings.warn("np.vdot() is faster on contiguous arrays, called on %s"
                              % ((a, b),), PerformanceWarning)
            if not all(x.dtype == a.dtype for x in (a, b)):
                raise TypingError("np.vdot() arguments must all have "
                                  "the same dtype")
            if not isinstance(a.dtype, (types.Float, types.Complex)):
                raise TypingError("np.vdot() only supported on "
                                  "float and complex arrays")
            return a.dtype

        return typer

if utils.HAS_MATMUL_OPERATOR:
    @infer_global(operator.matmul)
    class MatMul(MatMulTyperMixin, AbstractTemplate):
        key = operator.matmul
        func_name = "'@'"

        def generic(self, args, kws):
            assert not kws
            restype = self.matmul_typer(*args)
            if restype is not None:
                return signature(restype, *args)


def _check_linalg_matrix(a, func_name):
    if not isinstance(a, types.Array):
        return
    if not a.ndim == 2:
        raise TypingError("np.linalg.%s() only supported on 2-D arrays"
                          % func_name)
    if not isinstance(a.dtype, (types.Float, types.Complex)):
        raise TypingError("np.linalg.%s() only supported on "
                          "float and complex arrays" % func_name)

# -----------------------------------------------------------------------------
# Miscellaneous functions

@infer_global(np.ndenumerate)
class NdEnumerate(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        arr, = args

        if isinstance(arr, types.Array):
            enumerate_type = types.NumpyNdEnumerateType(arr)
            return signature(enumerate_type, *args)


@infer_global(np.nditer)
class NdIter(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        if len(args) != 1:
            return
        arrays, = args

        if isinstance(arrays, types.BaseTuple):
            if not arrays:
                return
            arrays = list(arrays)
        else:
            arrays = [arrays]
        nditerty = types.NumpyNdIterType(arrays)
        return signature(nditerty, *args)


@infer_global(pndindex)
@infer_global(np.ndindex)
class NdIndex(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws

        # Either ndindex(shape) or ndindex(*shape)
        if len(args) == 1 and isinstance(args[0], types.BaseTuple):
            tup = args[0]
            if tup.count > 0 and not isinstance(tup, types.UniTuple):
                # Heterogeneous tuple
                return
            shape = list(tup)
        else:
            shape = args

        if all(isinstance(x, types.Integer) for x in shape):
            iterator_type = types.NumpyNdIndexType(len(shape))
            return signature(iterator_type, *args)


# We use the same typing key for np.round() and np.around() to
# re-use the implementations automatically.
@infer_global(np.round)
@infer_global(np.around, typing_key=np.round)
class Round(AbstractTemplate):

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


@infer_global(np.where)
class Where(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws

        if len(args) == 1:
            # 0-dim arrays return one result array
            ary = args[0]
            ndim = max(ary.ndim, 1)
            retty = types.UniTuple(types.Array(types.intp, 1, 'C'), ndim)
            return signature(retty, ary)

        elif len(args) == 3:
            cond, x, y = args
            retdty = from_dtype(np.promote_types(
                        as_dtype(getattr(args[1], 'dtype', args[1])),
                        as_dtype(getattr(args[2], 'dtype', args[2]))))
            if isinstance(cond, types.Array):
                # array where()
                if (cond.ndim == x.ndim == y.ndim):
                    if x.layout == y.layout == cond.layout:
                        retty = types.Array(retdty, x.ndim, x.layout)
                    else:
                        retty = types.Array(retdty, x.ndim, 'C')
                    return signature(retty, *args)
            else:
                # scalar where()
                if not isinstance(x, types.Array):
                    retty = types.Array(retdty, 0, 'C')
                    return signature(retty, *args)


@infer_global(np.sinc)
class Sinc(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        arg = args[0]
        supported_scalars = (types.Float, types.Complex)
        if (isinstance(arg, supported_scalars) or
              (isinstance(arg, types.Array) and
               isinstance(arg.dtype, supported_scalars))):
            return signature(arg, arg)


@infer_global(np.angle)
class Angle(CallableTemplate):
    """
    Typing template for np.angle()
    """
    def generic(self):
        def typer(z, deg=False):
            if isinstance(z, types.Array):
                dtype = z.dtype
            else:
                dtype = z
            if isinstance(dtype, types.Complex):
                ret_dtype = dtype.underlying_float
            elif isinstance(dtype, types.Float):
                ret_dtype = dtype
            else:
                return
            if isinstance(z, types.Array):
                return z.copy(dtype=ret_dtype)
            else:
                return ret_dtype
        return typer


@infer_global(np.diag)
class DiagCtor(CallableTemplate):
    """
    Typing template for np.diag()
    """
    def generic(self):
        def typer(ref, k=0):
            if isinstance(ref, types.Array):
                if ref.ndim == 1:
                    rdim = 2
                elif ref.ndim == 2:
                    rdim = 1
                else:
                    return None
                if isinstance(k, (int, types.Integer)):
                    return types.Array(ndim=rdim, dtype=ref.dtype, layout='C')
        return typer


@infer_global(np.take)
class Take(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2
        arr, ind = args
        if isinstance(ind, types.Number):
            retty = arr.dtype
        elif isinstance(ind, types.Array):
            retty = types.Array(ndim=ind.ndim, dtype=arr.dtype, layout='C')
        elif isinstance(ind, types.List):
            retty = types.Array(ndim=1, dtype=arr.dtype, layout='C')
        elif isinstance(ind, types.BaseTuple):
            retty = types.Array(ndim=np.ndim(ind), dtype=arr.dtype, layout='C')
        else:
            return None

        return signature(retty, *args)

# -----------------------------------------------------------------------------
# Numba helpers

@infer_global(carray)
class NumbaCArray(CallableTemplate):
    layout = 'C'

    def generic(self):
        func_name = self.key.__name__

        def typer(ptr, shape, dtype=types.none):
            if ptr is types.voidptr:
                ptr_dtype = None
            elif isinstance(ptr, types.CPointer):
                ptr_dtype = ptr.dtype
            else:
                raise TypeError("%s(): pointer argument expected, got '%s'"
                                % (func_name, ptr))

            if dtype is types.none:
                if ptr_dtype is None:
                    raise TypeError("%s(): explicit dtype required for void* argument"
                                    % (func_name,))
                dtype = ptr_dtype
            elif isinstance(dtype, types.DTypeSpec):
                dtype = dtype.dtype
                if ptr_dtype is not None and dtype != ptr_dtype:
                    raise TypeError("%s(): mismatching dtype '%s' for pointer type '%s'"
                                    % (func_name, dtype, ptr))
            else:
                raise TypeError("%s(): invalid dtype spec '%s'"
                                % (func_name, dtype))

            ndim = _parse_shape(shape)
            if ndim is None:
                raise TypeError("%s(): invalid shape '%s'"
                                % (func_name, shape))

            return types.Array(dtype, ndim, self.layout)

        return typer


@infer_global(farray)
class NumbaFArray(NumbaCArray):
    layout = 'F'
