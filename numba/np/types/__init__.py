import numpy as np

from numba import types
from .datetime import NPDatetime, NPTimedelta

numpy_version = tuple(map(int, np.__version__.split('.')[:2]))

if numpy_version >= (2, 0):
    types.bool = types.bool_
    # types.__all__.append('bool')

if numpy_version < (2, 0):
    types.float_ = types.float32
    types.__all__.append('float_')

from numba.np.types.datetime import NPDatetime, NPTimedelta
