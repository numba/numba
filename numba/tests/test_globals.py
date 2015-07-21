from __future__ import print_function, division, absolute_import
import numpy as np
from numba import jit
from numba import unittest_support as unittest
from numba.tests import usecases

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
x_dt = np.dtype([('a', np.int32), ('b', np.float32)])
rec_X = np.recarray(10, dtype=x_dt)
for i in range(len(rec_X)):
    rec_X[i].a = i
    rec_X[i].b = i + 0.5


def global_rec_arr_copy(a):
    for i in range(len(a)):
        a[i] = rec_X[i]


def global_rec_arr_extract_fields(a, b):
    for i in range(len(a)):
        a[i] = rec_X[i].a
        b[i] = rec_X[i].b


# Create additional global recarray
y_dt = np.dtype([('c', np.int16), ('d', np.float64)])
rec_Y = np.recarray(10, dtype=y_dt)
for i in range(len(rec_Y)):
    rec_Y[i].c = i + 10
    rec_Y[i].d = i + 10.5


def global_two_rec_arrs(a, b, c, d):
    for i in range(len(a)):
        a[i] = rec_X[i].a
        b[i] = rec_X[i].b
        c[i] = rec_Y[i].c
        d[i] = rec_Y[i].d


# Test a global record
record_only_X = np.recarray(1, dtype=x_dt)[0]
record_only_X.a = 1
record_only_X.b = 1.5

@jit(nopython=True)
def global_record_func(x):
    return x.a == record_only_X.a


@jit(nopython=True)
def global_module_func(x, y):
    return usecases.andornopython(x, y)


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
        arr = np.zeros(len(cplx_X), dtype=np.complex128)
        ctestfunc(arr)
        np.testing.assert_equal(arr, cplx_X)

    def test_global_complex_arr(self):
        self.check_global_complex_arr(forceobj=True)

    def test_global_complex_arr_npm(self):
        self.check_global_complex_arr(nopython=True)


    def check_global_rec_arr(self, **jitargs):
        # (see github issue #897)
        ctestfunc = jit(**jitargs)(global_rec_arr_copy)
        arr = np.zeros(rec_X.shape, dtype=x_dt)
        ctestfunc(arr)
        np.testing.assert_equal(arr, rec_X)

    def test_global_rec_arr(self):
        self.check_global_rec_arr(forceobj=True)

    def test_global_rec_arr_npm(self):
        self.check_global_rec_arr(nopython=True)


    def check_global_rec_arr_extract(self, **jitargs):
        # (see github issue #897)
        ctestfunc = jit(**jitargs)(global_rec_arr_extract_fields)
        arr1 = np.zeros(rec_X.shape, dtype=np.int32)
        arr2 = np.zeros(rec_X.shape, dtype=np.float32)
        ctestfunc(arr1, arr2)
        np.testing.assert_equal(arr1, rec_X.a)
        np.testing.assert_equal(arr2, rec_X.b)

    def test_global_rec_arr_extract(self):
        self.check_global_rec_arr_extract(forceobj=True)

    def test_global_rec_arr_extract_npm(self):
        self.check_global_rec_arr_extract(nopython=True)


    def check_two_global_rec_arrs(self, **jitargs):
        # (see github issue #897)
        ctestfunc = jit(**jitargs)(global_two_rec_arrs)
        arr1 = np.zeros(rec_X.shape, dtype=np.int32)
        arr2 = np.zeros(rec_X.shape, dtype=np.float32)
        arr3 = np.zeros(rec_Y.shape, dtype=np.int16)
        arr4 = np.zeros(rec_Y.shape, dtype=np.float64)
        ctestfunc(arr1, arr2, arr3, arr4)
        np.testing.assert_equal(arr1, rec_X.a)
        np.testing.assert_equal(arr2, rec_X.b)
        np.testing.assert_equal(arr3, rec_Y.c)
        np.testing.assert_equal(arr4, rec_Y.d)

    def test_two_global_rec_arrs(self):
        self.check_two_global_rec_arrs(forceobj=True)

    def test_two_global_rec_arrs_npm(self):
        self.check_two_global_rec_arrs(nopython=True)

    def test_global_module(self):
        # (see github issue #1059)
        res = global_module_func(5, 6)
        self.assertEqual(True, res)

    def test_global_record(self):
        # (see github issue #1081)
        x = np.recarray(1, dtype=x_dt)[0]
        x.a = 1
        res = global_record_func(x)
        self.assertEqual(True, res)
        x.a = 2
        res = global_record_func(x)
        self.assertEqual(False, res)

if __name__ == '__main__':
    unittest.main()
