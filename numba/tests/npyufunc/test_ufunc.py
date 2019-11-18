from __future__ import absolute_import, print_function, division

import numpy as np

from numba import unittest_support as unittest
from numba import float32, jit
from numba.npyufunc import Vectorize
from numba.errors import TypingError
from ..support import tag, TestCase


dtype = np.float32
a = np.arange(80, dtype=dtype).reshape(8, 10)
b = a.copy()
c = a.copy(order='F')
d = np.arange(16 * 20, dtype=dtype).reshape(16, 20)[::2, ::2]


def add(a, b):
    return a + b


def add_multiple_args(a, b, c, d):
    return a + b + c + d


def gufunc_add(a, b):
    result = 0.0
    for i in range(a.shape[0]):
        result += a[i] * b[i]

    return result


def ufunc_reduce(ufunc, arg):
    for i in range(arg.ndim):
        arg = ufunc.reduce(arg)
    return arg


vectorizers = [
    Vectorize,
    # ParallelVectorize,
    # StreamVectorize,
    # CudaVectorize,
    # GUFuncVectorize,
]


class TestUFuncs(TestCase):

    def _test_ufunc_attributes(self, cls, a, b, *args):
        "Test ufunc attributes"
        vectorizer = cls(add, *args)
        vectorizer.add(float32(float32, float32))
        ufunc = vectorizer.build_ufunc()

        info = (cls, a.ndim)
        self.assertPreciseEqual(ufunc(a, b), a + b, msg=info)
        self.assertPreciseEqual(ufunc_reduce(ufunc, a), np.sum(a), msg=info)
        self.assertPreciseEqual(ufunc.accumulate(a), np.add.accumulate(a),
                                msg=info)
        self.assertPreciseEqual(ufunc.outer(a, b), np.add.outer(a, b), msg=info)

    def _test_broadcasting(self, cls, a, b, c, d):
        "Test multiple args"
        vectorizer = cls(add_multiple_args)
        vectorizer.add(float32(float32, float32, float32, float32))
        ufunc = vectorizer.build_ufunc()

        info = (cls, a.shape)
        self.assertPreciseEqual(ufunc(a, b, c, d), a + b + c + d, msg=info)

    @tag('important')
    def test_ufunc_attributes(self):
        for v in vectorizers: # 1D
            self._test_ufunc_attributes(v, a[0], b[0])
        for v in vectorizers: # 2D
            self._test_ufunc_attributes(v, a, b)
        for v in vectorizers: # 3D
            self._test_ufunc_attributes(v, a[:, np.newaxis, :],
                                        b[np.newaxis, :, :])

    @tag('important')
    def test_broadcasting(self):
        for v in vectorizers: # 1D
            self._test_broadcasting(v, a[0], b[0], c[0], d[0])
        for v in vectorizers: # 2D
            self._test_broadcasting(v, a, b, c, d)
        for v in vectorizers: # 3D
            self._test_broadcasting(v, a[:, np.newaxis, :], b[np.newaxis, :, :],
                                    c[:, np.newaxis, :], d[np.newaxis, :, :])

    @tag('important')
    def test_implicit_broadcasting(self):
        for v in vectorizers:
            vectorizer = v(add)
            vectorizer.add(float32(float32, float32))
            ufunc = vectorizer.build_ufunc()

            broadcasting_b = b[np.newaxis, :, np.newaxis, np.newaxis, :]
            self.assertPreciseEqual(ufunc(a, broadcasting_b),
                                    a + broadcasting_b)

    def test_ufunc_exception_on_write_to_readonly(self):
        z = np.ones(10)
        z.flags.writeable = False # flip write bit

        tests = []
        expect = "ufunc 'sin' called with an explicit output that is read-only"
        tests.append((jit(nopython=True), TypingError, expect))
        tests.append((jit(nopython=False), ValueError,
                      "output array is read-only"))

        for dec, exc, msg in tests:
            def test(x):
                a = np.ones(x.shape, x.dtype) # do not copy RO attribute from x
                np.sin(a, x)

            with self.assertRaises(exc) as raises:
                dec(test)(z)

            self.assertIn(msg, str(raises.exception))


if __name__ == '__main__':
    unittest.main()
