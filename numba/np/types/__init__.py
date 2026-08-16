import numpy as np
import sys
import inspect
from numba import types
from .datetime import NPDatetime, NPTimedelta
from .npytypes import *

numpy_version = tuple(map(int, np.__version__.split('.')[:2]))

if numpy_version >= (2, 0):
    types.bool = types.bool_
    # types.__all__.append('bool')

if numpy_version < (2, 0):
    types.float_ = types.float32
    # types.__all__.append('float_')

npy_rng = NumPyRandomGeneratorType('rng')
npy_bitgen = NumPyRandomBitGeneratorType('bitgen')

# Explicitly register these as attributes of the types module
_exported = {
    "NPDatetime": NPDatetime,
    "NPTimedelta": NPTimedelta,
    "Array": Array,
    "ArrayCTypes": ArrayCTypes,
    "ArrayFlags": ArrayFlags,
    "DType": DType,
    "NestedArray": NestedArray,
    "NumpyNdIterType": NumpyNdIterType,
    "NumpyNdEnumerateType": NumpyNdEnumerateType,
    "NumPyRandomGeneratorType": NumPyRandomGeneratorType,
    "NumPyRandomBitGeneratorType": NumPyRandomBitGeneratorType,
    "Record": Record,
    "NumpyFlatType": NumpyFlatType,
    "NumpyNdIndexType": NumpyNdIndexType,
    "PolynomialType": PolynomialType,
    "npy_rng": npy_rng,
    "npy_bitgen": npy_bitgen,
}

for _name, _obj in _exported.items():
    setattr(types, _name, _obj)
