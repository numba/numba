import numpy as np
from numpy.random.bit_generator import BitGenerator


from numba.core import types, errors
from numba.core.typing.typeof import typeof_impl, typeof, _extra_types
from numba.np import numpy_support


@typeof_impl.register(np.generic)
def _typeof_numpy_scalar(val, c):
    try:
        return numpy_support.map_arrayscalar_type(val)
    except errors.NumbaNotImplementedError:
        pass
    except NotImplementedError:
        pass


@typeof_impl.register(np.ndarray)
def _typeof_ndarray(val, c):
    if isinstance(val, np.ma.MaskedArray):
        msg = "Unsupported array type: numpy.ma.MaskedArray."
        raise errors.NumbaTypeError(msg)
    try:
        dtype = numpy_support.from_dtype(val.dtype)
    except errors.NumbaNotImplementedError:
        raise errors.NumbaValueError(f"Unsupported array dtype: {val.dtype}")
    layout = numpy_support.map_layout(val)
    readonly = not val.flags.writeable
    return types.Array(dtype, val.ndim, layout, readonly=readonly)


@typeof_impl.register(np.dtype)
def _typeof_dtype(val, c):
    tp = numpy_support.from_dtype(val)
    return types.DType(tp)


@typeof_impl.register(BitGenerator)
def typeof_numpy_random_bitgen(val, c):
    return types.NumPyRandomBitGeneratorType(val)


@typeof_impl.register(np.random.Generator)
def typeof_random_generator(val, c):
    return types.NumPyRandomGeneratorType(val)


@typeof_impl.register(np.polynomial.polynomial.Polynomial)
def typeof_numpy_polynomial(val, c):
    coef = typeof(val.coef)
    domain = typeof(val.domain)
    window = typeof(val.window)
    return types.PolynomialType(coef, domain, window)


@typeof_impl.register(types.NumberClass)
def _typeof_number_class(val, c):
    return val


_extra_types[np.generic] = lambda x: types.NumberClass(
    numpy_support.from_dtype(x)
)
