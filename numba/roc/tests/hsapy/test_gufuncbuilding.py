import numpy as np

from numba.roc.vectorizers import HsaGUFuncVectorize
from numba.roc.dispatch import HSAGenerializedUFunc
from numba import guvectorize
import unittest


def ufunc_add_core(a, b, c):
    for i in range(c.size):
        c[i] = a[i] + b[i]


class TestGUFuncBuilding(unittest.TestCase):
    def test_gufunc_building(self):
        ufbldr = HsaGUFuncVectorize(ufunc_add_core, "(x),(x)->(x)")
        ufbldr.add("(float32[:], float32[:], float32[:])")
        ufbldr.add("(intp[:], intp[:], intp[:])")
        ufunc = ufbldr.build_ufunc()
        self.assertIsInstance(ufunc, HSAGenerializedUFunc)

        # Test integer version
        A = np.arange(100, dtype=np.intp)
        B = np.arange(100, dtype=np.intp) + 1
        expected = A + B
        got = ufunc(A, B)

        np.testing.assert_equal(expected, got)
        self.assertEqual(expected.dtype, got.dtype)
        self.assertEqual(np.dtype(np.intp), got.dtype)

        # Test integer version with 2D inputs
        A = A.reshape(50, 2)
        B = B.reshape(50, 2)
        expected = A + B
        got = ufunc(A, B)

        np.testing.assert_equal(expected, got)
        self.assertEqual(expected.dtype, got.dtype)
        self.assertEqual(np.dtype(np.intp), got.dtype)

        # Test integer version with 3D inputs
        A = A.reshape(5, 10, 2)
        B = B.reshape(5, 10, 2)
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

        # Test real version with 2D inputs
        A = A.reshape(50, 2)
        B = B.reshape(50, 2)
        expected = A + B
        got = ufunc(A, B)

        np.testing.assert_allclose(expected, got)
        self.assertEqual(expected.dtype, got.dtype)
        self.assertEqual(np.dtype(np.float32), got.dtype)

    def test_gufunc_building_scalar_output(self):
        def sum_row(inp, out):
            tmp = 0.
            for i in range(inp.shape[0]):
                tmp += inp[i]
            out[0] = tmp

        ufbldr = HsaGUFuncVectorize(sum_row, "(n)->()")
        ufbldr.add("void(int32[:], int32[:])")
        ufunc = ufbldr.build_ufunc()

        inp = np.arange(300, dtype=np.int32).reshape(100, 3)
        out = ufunc(inp)

        for i in range(inp.shape[0]):
            np.testing.assert_equal(inp[i].sum(), out[i])

    def test_gufunc_scalar_input_saxpy(self):
        def axpy(a, x, y, out):
            for i in range(out.shape[0]):
                out[i] = a * x[i] + y[i]

        ufbldr = HsaGUFuncVectorize(axpy, '(),(t),(t)->(t)')
        ufbldr.add("void(float32, float32[:], float32[:], float32[:])")
        saxpy = ufbldr.build_ufunc()

        A = np.float32(2)
        X = np.arange(10, dtype=np.float32).reshape(5, 2)
        Y = np.arange(10, dtype=np.float32).reshape(5, 2)
        out = saxpy(A, X, Y)

        for j in range(5):
            for i in range(2):
                exp = A * X[j, i] + Y[j, i]
                self.assertTrue(exp == out[j, i])

        X = np.arange(10, dtype=np.float32)
        Y = np.arange(10, dtype=np.float32)
        out = saxpy(A, X, Y)

        for j in range(10):
            exp = A * X[j] + Y[j]
            self.assertTrue(exp == out[j], (exp, out[j]))

        A = np.arange(5, dtype=np.float32)
        X = np.arange(10, dtype=np.float32).reshape(5, 2)
        Y = np.arange(10, dtype=np.float32).reshape(5, 2)
        out = saxpy(A, X, Y)

        for j in range(5):
            for i in range(2):
                exp = A[j] * X[j, i] + Y[j, i]
                self.assertTrue(exp == out[j, i], (exp, out[j, i]))


class TestGUFuncDecor(unittest.TestCase):
    def test_gufunc_decorator(self):
        @guvectorize(["void(float32, float32[:], float32[:], float32[:])"],
                     '(),(t),(t)->(t)', target='roc')
        def saxpy(a, x, y, out):
            for i in range(out.shape[0]):
                out[i] = a * x[i] + y[i]

        A = np.float32(2)
        X = np.arange(10, dtype=np.float32).reshape(5, 2)
        Y = np.arange(10, dtype=np.float32).reshape(5, 2)
        out = saxpy(A, X, Y)

        for j in range(5):
            for i in range(2):
                exp = A * X[j, i] + Y[j, i]
                self.assertTrue(exp == out[j, i])

        X = np.arange(10, dtype=np.float32)
        Y = np.arange(10, dtype=np.float32)
        out = saxpy(A, X, Y)

        for j in range(10):
            exp = A * X[j] + Y[j]
            self.assertTrue(exp == out[j], (exp, out[j]))

        A = np.arange(5, dtype=np.float32)
        X = np.arange(10, dtype=np.float32).reshape(5, 2)
        Y = np.arange(10, dtype=np.float32).reshape(5, 2)
        out = saxpy(A, X, Y)

        for j in range(5):
            for i in range(2):
                exp = A[j] * X[j, i] + Y[j, i]
                self.assertTrue(exp == out[j, i], (exp, out[j, i]))


if __name__ == '__main__':
    unittest.main()
