from __future__ import print_function, division, absolute_import
import numpy as np
from numba import jit
from numba import unittest_support as unittest


X = np.arange(10)


def global_ndarray_func(x):
    y = x + X.shape[0]
    return y


# Create complex array with real and imaginary parts of distinct value
cplx_X = np.arange(10, dtype=np.complex128)
tmp = np.arange(10, dtype=np.complex128)
cplx_X += (tmp+10)*1j


def global_cplx_arr_copy(a):
    for i in range(len(a)):
        a[i] = cplx_X[i]


# Create a recarray with fields of distinct value
rec_X = np.recarray(10, dtype=np.dtype([('a', np.int32), ('b', np.float32)]))
for i in range(len(rec_X)):
    rec_X[i]['a'] = i
    rec_X[i]['b'] = i + 0.5


def global_rec_arr_copy(a):
    for i in range(len(a)):
        a[i] = rec_X[i]


class TestGlobals(unittest.TestCase):

    def check_global_ndarray(self, **jitargs):
        # (see github issue #448)
        ctestfunc = jit(**jitargs)(global_ndarray_func)
        self.assertEqual(ctestfunc(1), 11)

    def test_global_ndarray(self):
        # This also checks we can access an unhashable global value
        # (see issue #697)
        self.check_global_ndarray(forceobj=True)

    def test_global_ndarray_npm(self):
        self.check_global_ndarray(nopython=True)


    def check_global_complex_arr(self, **jitargs):
        # (see github issue #897)
        ctestfunc = jit(**jitargs)(global_cplx_arr_copy)
        arr = np.zeros_like(cplx_X)
        ctestfunc(arr)
        np.testing.assert_equal(arr, cplx_X)

    def test_global_complex_arr(self):
        self.check_global_complex_arr(forceobj=True)

    def test_global_complex_arr_npm(self):
        self.check_global_complex_arr(nopython=True)


    def check_global_rec_arr(self, **jitargs):
        # (see github issue #897)
        ctestfunc = jit(**jitargs)(global_rec_arr_copy)
        arr = np.zeros_like(rec_X)
        ctestfunc(arr)
        np.testing.assert_equal(arr, rec_X)

    def test_global_complex_arr(self):
        self.check_global_rec_arr(forceobj=True)

    def test_global_complex_arr_npm(self):
        self.check_global_rec_arr(nopython=True)


if __name__ == '__main__':
    unittest.main()
