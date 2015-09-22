from __future__ import absolute_import, print_function, division

import numpy as np

from numba import unittest_support as unittest
from numba import vectorize, guvectorize

from ..support import TestCase


def sqrt(val):
    if val < 0.0:
        raise ValueError('Value must be positive')
    return val ** 0.5


def gufunc_foo(inp, n, out):
    for i in range(inp.shape[0]):
        if inp[i] < 0:
            raise ValueError('Value must be positive')
        out[i] = inp[i] * n[0]


class TestExceptions(TestCase):
    """
    Test raising exceptions inside ufuncs.
    """

    def check_ufunc_raise(self, **vectorize_args):
        f = vectorize(['float64(float64)'], **vectorize_args)(sqrt)
        arr = np.array([1, 4, -2, 9, -1, 16], dtype=np.float64)
        out = np.zeros_like(arr)
        with self.assertRaises(ValueError) as cm:
            f(arr, out)
        self.assertIn('Value must be positive', str(cm.exception))
        # All values were computed except for the ones giving an error
        self.assertEqual(list(out), [1, 2, 0, 3, 0, 4])

    def test_ufunc_raise(self):
        self.check_ufunc_raise(nopython=True)

    def test_ufunc_raise_objmode(self):
        self.check_ufunc_raise(forceobj=True)

    def check_gufunc_raise(self, **vectorize_args):
        f = guvectorize(['int32[:], int32[:], int32[:]'], '(n),()->(n)',
                        **vectorize_args)(gufunc_foo)
        arr = np.array([1, 2, -3, 4], dtype=np.int32)
        out = np.zeros_like(arr)
        with self.assertRaises(ValueError) as cm:
            f(arr, 2, out)
        # The gufunc bailed out after the error
        self.assertEqual(list(out), [2, 4, 0, 0])

    def test_gufunc_raise(self):
        self.check_gufunc_raise(nopython=True)

    def test_gufunc_raise_objmode(self):
        self.check_gufunc_raise(forceobj=True)


if __name__ == "__main__":
    unittest.main()
