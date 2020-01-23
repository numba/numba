import sys

try:
    import scipy.linalg.cython_blas
    has_blas = True
except ImportError:
    has_blas = False

import numba.unittest_support as unittest
from numba.numpy_support import version as numpy_version

def matmul_usecase(x, y):
    return x @ y

def imatmul_usecase(x, y):
    x @= y
    return x

needs_blas = unittest.skipUnless(has_blas, "BLAS needs SciPy 1.0+")

class DumbMatrix(object):

    def __init__(self, value):
        self.value = value

    def __matmul__(self, other):
        if isinstance(other, DumbMatrix):
            return DumbMatrix(self.value * other.value)
        return NotImplemented

    def __imatmul__(self, other):
        if isinstance(other, DumbMatrix):
            self.value *= other.value
            return self
        return NotImplemented
