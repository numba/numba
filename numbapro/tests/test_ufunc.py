import unittest

import numpy as np

from numba import *
from numbapro.vectorize import *

dtype = np.float32
a = np.arange(80, dtype=dtype).reshape(8, 10)
b = a.copy()
c = a.copy(order='F')
d = np.arange(16 * 20, dtype=dtype).reshape(16, 20)[::2, ::2]

def add(a, b):
    return a + b

def add_multiple_args(a, b, c, d):
    return a + b + c + d

vectorizers = [
    BasicVectorize,
    ParallelVectorize,
    StreamVectorize,
    CudaVectorize,
    # MiniVectorize,
    GUFuncVectorize,
]

def ufunc_reduce(ufunc, arg):
    for i in range(arg.ndim):
        arg = ufunc.reduce(arg)
    return arg

class TestUFuncs(object): #unittest.TestCase):
    def _test_ufunc_attributes(self, cls, a, b):
        "Test ufunc attributes"
        vectorizer = cls(add)
        vectorizer.add(ret_type=f, arg_types=[f, f])
        ufunc = vectorizer.build_ufunc()

        assert np.all(ufunc(a, b) == a + b), ufunc(a, b)
        assert ufunc_reduce(ufunc, a) == np.sum(a)
        assert np.all(ufunc.accumulate(a) == np.add.accumulate(a))
        assert np.all(ufunc.outer(a, b) == np.add.outer(a, b))

    def _test_multiple_args(self, cls, a, b, c, d):
        "Test multiple args"
        vectorizer = cls(add_multiple_args)
        vectorizer.add(ret_type=f, arg_types=[f, f, f, f])
        ufunc = vectorizer.build_ufunc()

        print ufunc(a, b, c, d)
        assert np.all(ufunc(a, b, c, d) == a + b + c + d)

    def test_ufunc_attributes(self):
        for v in vectorizers: # 1D
            self._test_ufunc_attributes(v, a[0], b[0])
        for v in vectorizers: # 2D
            self._test_ufunc_attributes(v, a, b)
        for v in vectorizers: # 3D
            self._test_ufunc_attributes(v, a[:, np.newaxis, :],
                                           b[np.newaxis, :, :])

    def test_multiple_args(self):
        for v in vectorizers: # 1D
            self._test_multiple_args(v, a[0], b[0], c[0], d[0])
        for v in vectorizers: # 2D
            self._test_multiple_args(v, a, b, c, d)
        for v in vectorizers: # 3D
            self._test_multiple_args(v, a[:, np.newaxis, :], b[np.newaxis, :, :],
                                        c[:, :], d[np.newaxis, :, :])


TestUFuncs().test_multiple_args()
if __name__ == '__main__':
    unittest.main()