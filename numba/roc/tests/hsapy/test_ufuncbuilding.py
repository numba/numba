from __future__ import print_function, absolute_import

import numpy as np

from numba import unittest_support as unittest
from numba import vectorize
from numba.roc.vectorizers import HsaVectorize
from numba.roc.dispatch import HsaUFuncDispatcher


def ufunc_add_core(a, b):
    return a + b


class TestUFuncBuilding(unittest.TestCase):
    def test_ufunc_building(self):
        ufbldr = HsaVectorize(ufunc_add_core)
        ufbldr.add("float32(float32, float32)")
        ufbldr.add("intp(intp, intp)")
        ufunc = ufbldr.build_ufunc()
        self.assertIsInstance(ufunc, HsaUFuncDispatcher)

        # Test integer version
        A = np.arange(100, dtype=np.intp)
        B = np.arange(100, dtype=np.intp) + 1
        expected = A + B
        got = ufunc(A, B)

        np.testing.assert_equal(expected, got)
        self.assertEqual(expected.dtype, got.dtype)
        self.assertEqual(np.dtype(np.intp), got.dtype)

        # Test real version
        A = np.arange(100, dtype=np.float32)
        B = np.arange(100, dtype=np.float32) + 1
        expected = A + B
        got = ufunc(A, B)

        np.testing.assert_allclose(expected, got)
        self.assertEqual(expected.dtype, got.dtype)
        self.assertEqual(np.dtype(np.float32), got.dtype)


class TestVectorizeDecor(unittest.TestCase):
    def test_vectorize_decor(self):
        @vectorize(["float32(float32, float32, float32)",
                    "intp(intp, intp, intp)"],
                   target='roc')
        def axpy(a, x, y):
            return a * x + y


        self.assertIsInstance(axpy, HsaUFuncDispatcher)
        # Test integer version
        A = np.arange(100, dtype=np.intp)
        X = np.arange(100, dtype=np.intp) + 1
        Y = np.arange(100, dtype=np.intp) + 2
        expected = A * X + Y
        got = axpy(A, X, Y)

        np.testing.assert_equal(expected, got)
        self.assertEqual(expected.dtype, got.dtype)
        self.assertEqual(np.dtype(np.intp), got.dtype)

        # Test real version
        A = np.arange(100, dtype=np.float32)
        X = np.arange(100, dtype=np.float32) + 1
        Y = np.arange(100, dtype=np.float32) + 2
        expected = A * X + Y
        got = axpy(A, X, Y)

        np.testing.assert_allclose(expected, got)
        self.assertEqual(expected.dtype, got.dtype)
        self.assertEqual(np.dtype(np.float32), got.dtype)


class TestVectorizeScalar(unittest.TestCase):
    def test_scalar_input(self):
        @vectorize(["float32(float32, float32, float32)",
                    "intp(intp, intp, intp)"],
                   target='roc')
        def axpy(a, x, y):
            return a * x + y

        self.assertIsInstance(axpy, HsaUFuncDispatcher)
        # Test integer version
        A = 2
        X = np.arange(100, dtype=np.intp) + 1
        Y = np.arange(100, dtype=np.intp) + 2
        expected = A * X + Y
        got = axpy(A, X, Y)

        np.testing.assert_equal(expected, got)
        self.assertEqual(expected.dtype, got.dtype)
        self.assertEqual(np.dtype(np.intp), got.dtype)

        # Test real version
        A = 2.3
        X = np.arange(100, dtype=np.float32) + 1
        Y = np.arange(100, dtype=np.float32) + 2
        expected = A * X + Y
        got = axpy(A, X, Y)

        np.testing.assert_allclose(expected, got)
        self.assertEqual(expected.dtype, got.dtype)
        self.assertEqual(np.dtype(np.float32), got.dtype)


if __name__ == '__main__':
    unittest.main()
