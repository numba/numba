import numpy as np

from numba import types
from .datetime import NPDatetime, NPTimedelta

numpy_version = tuple(map(int, np.__version__.split('.')[:2]))

if numpy_version >= (2, 0):
    types.bool = types.bool_

if numpy_version < (2, 0):
    types.float_ = types.float32

if numpy_version >= (2, 0):
    types.__all__.remove('float_')
    types.__all__.append('bool')

from numba.np.types.datetime import NPDatetime, NPTimedelta
