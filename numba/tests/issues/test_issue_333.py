# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from numba import jit, int_
import numpy as np

@jit(argtypes=(int_[:, :], int_[:]), nopython=False)
def npsum(x, y):
    return np.sum(x, 0) + y

result = npsum(np.arange(9).reshape((3,3)), np.array([1,2,3]))
assert np.array_equal(result, np.array([10, 14, 18])) == True

