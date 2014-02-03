from numba import *
from numba import vectorize, typeof

import numpy as np

@vectorize.vectorize([
    bool_(double, int_),
    double(double, double),
    float_(double, float_),
])
def add(a, b):
    return a + b

@autojit
def func(dtypeA, dtypeB):
    A = np.arange(10, dtype=dtypeA)
    B = np.arange(10, dtype=dtypeB)
    return typeof(add(A, B))

if __name__ == "__main__":
    assert func(np.dtype(np.float64), np.dtype('i')) == int8[:]
    assert func(np.dtype(np.float64), np.dtype(np.float64)) == double[:]
    assert func(np.dtype(np.float64), np.dtype(np.float32)) == float_[:]
