#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import numpy as np

from numba import jit


record_type = np.dtype([('x', np.double), ('y', np.double)])
a = np.array([(1.0, 2.0), (3.0, 4.0)], dtype=record_type)

@jit
def hypot(data):
    result = np.empty_like(data, dtype=np.float64)
    # notice access to structure elements 'x' and 'y' via attribute access
    for i in range(data.shape[0]):
        result[i] = np.sqrt(data[i].x * data[i].x + data[i].y * data[i].y)

    return result


print(hypot(a))
