from __future__ import print_function, division, absolute_import
import numpy
from numba import types, config

version = tuple(map(int, numpy.__version__.split('.')[:2]))
int_divbyzero_returns_zero = config.PYVERSION <= (3, 0)


FROM_DTYPE = {
    numpy.dtype('bool'): types.boolean,
    numpy.dtype('int8'): types.int8,
    numpy.dtype('int16'): types.int16,
    numpy.dtype('int32'): types.int32,
    numpy.dtype('int64'): types.int64,

    numpy.dtype('uint8'): types.uint8,
    numpy.dtype('uint16'): types.uint16,
    numpy.dtype('uint32'): types.uint32,
    numpy.dtype('uint64'): types.uint64,

    numpy.dtype('float32'): types.float32,
    numpy.dtype('float64'): types.float64,

    numpy.dtype('complex64'): types.complex64,
    numpy.dtype('complex128'): types.complex128,
}


def from_dtype(dtype):
    return FROM_DTYPE[dtype]


def is_scalar_array(val):
    return numpy.dtype(type(val)) in FROM_DTYPE

