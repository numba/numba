from __future__ import division

from itertools import product
import sys

import numpy as np

from numba import unittest_support as unittest
from numba import jit, errors
from .support import TestCase
from .matmul_usecase import matmul_usecase, needs_matmul, needs_blas


def dot2(a, b):
    return np.dot(a, b)

def dot3(a, b, out):
    return np.dot(a, b, out=out)

def vdot(a, b):
    return np.vdot(a, b)


class TestProduct(TestCase):
    """
    Tests for dot products.
    """

    dtypes = (np.float64, np.float32, np.complex128, np.complex64)

    def sample_vector(self, n, dtype):
        if issubclass(dtype, np.complexfloating):
            return np.linspace(-1j, 5 + 2j, n).astype(dtype)
        else:
            return np.linspace(-1, 5, n).astype(dtype)

    def sample_matrix(self, m, n, dtype):
        if issubclass(dtype, np.complexfloating):
            return np.linspace(-1j, 5 + 2j, m * n).reshape((m, n)).astype(dtype)
        else:
            return np.linspace(-1, 5, m * n).reshape((m, n)).astype(dtype)

    def check_func(self, pyfunc, cfunc, args):
        expected = pyfunc(*args)
        got = cfunc(*args)
        self.assertPreciseEqual(got, expected)

    def check_func_out(self, pyfunc, cfunc, args, out):
        expected = np.copy(out)
        got = np.copy(out)
        self.assertIs(pyfunc(*args, out=expected), expected)
        self.assertIs(cfunc(*args, out=got), got)
        self.assertPreciseEqual(got, expected)

    def assert_mismatching_sizes(self, cfunc, args, is_out=False):
        with self.assertRaises(ValueError) as raises:
            cfunc(*args)
        msg = ("incompatible output array size" if is_out else
               "incompatible array sizes")
        self.assertIn(msg, str(raises.exception))

    def assert_mismatching_dtypes(self, cfunc, args, func_name="np.dot()"):
        with self.assertRaises(errors.TypingError) as raises:
            cfunc(*args)
        self.assertIn("%s arguments must all have the same dtype"
                      % (func_name,),
                      str(raises.exception))

    @needs_blas
    def check_dot_vv(self, pyfunc, func_name):
        n = 3
        cfunc = jit(nopython=True)(pyfunc)
        for dtype in self.dtypes:
            a = self.sample_vector(n, dtype)
            b = self.sample_vector(n, dtype)
            self.check_func(pyfunc, cfunc, (a, b))

        # Mismatching sizes
        a = self.sample_vector(n - 1, np.float64)
        b = self.sample_vector(n, np.float64)
        self.assert_mismatching_sizes(cfunc, (a, b))
        # Mismatching dtypes
        a = self.sample_vector(n, np.float32)
        b = self.sample_vector(n, np.float64)
        self.assert_mismatching_dtypes(cfunc, (a, b), func_name=func_name)

    def test_dot_vv(self):
        """
        Test vector * vector np.dot()
        """
        self.check_dot_vv(dot2, "np.dot()")

    def test_vdot(self):
        """
        Test np.vdot()
        """
        self.check_dot_vv(vdot, "np.vdot()")

    @needs_blas
    def check_dot_vm(self, pyfunc2, pyfunc3, func_name):
        m, n = 2, 3

        def samples(m, n):
            for order in 'CF':
                a = self.sample_matrix(m, n, np.float64).copy(order=order)
                b = self.sample_vector(n, np.float64)
                yield a, b
            for dtype in self.dtypes:
                a = self.sample_matrix(m, n, dtype)
                b = self.sample_vector(n, dtype)
                yield a, b

        cfunc2 = jit(nopython=True)(pyfunc2)
        if pyfunc3 is not None:
            cfunc3 = jit(nopython=True)(pyfunc3)
        for a, b in samples(m, n):
            self.check_func(pyfunc2, cfunc2, (a, b))
            self.check_func(pyfunc2, cfunc2, (b, a.T))
        if pyfunc3 is not None:
            for a, b in samples(m, n):
                out = np.empty(m, dtype=a.dtype)
                self.check_func_out(pyfunc3, cfunc3, (a, b), out)
                self.check_func_out(pyfunc3, cfunc3, (b, a.T), out)

        # Mismatching sizes
        a = self.sample_matrix(m, n - 1, np.float64)
        b = self.sample_vector(n, np.float64)
        self.assert_mismatching_sizes(cfunc2, (a, b))
        self.assert_mismatching_sizes(cfunc2, (b, a.T))
        if pyfunc3 is not None:
            out = np.empty(m, np.float64)
            self.assert_mismatching_sizes(cfunc3, (a, b, out))
            self.assert_mismatching_sizes(cfunc3, (b, a.T, out))
            a = self.sample_matrix(m, m, np.float64)
            b = self.sample_vector(m, np.float64)
            out = np.empty(m - 1, np.float64)
            self.assert_mismatching_sizes(cfunc3, (a, b, out), is_out=True)
            self.assert_mismatching_sizes(cfunc3, (b, a.T, out), is_out=True)
        # Mismatching dtypes
        a = self.sample_matrix(m, n, np.float32)
        b = self.sample_vector(n, np.float64)
        self.assert_mismatching_dtypes(cfunc2, (a, b), func_name)
        if pyfunc3 is not None:
            a = self.sample_matrix(m, n, np.float64)
            b = self.sample_vector(n, np.float64)
            out = np.empty(m, np.float32)
            self.assert_mismatching_dtypes(cfunc3, (a, b, out), func_name)

    def test_dot_vm(self):
        """
        Test vector * matrix and matrix * vector np.dot()
        """
        self.check_dot_vm(dot2, dot3, "np.dot()")

    @needs_blas
    def check_dot_mm(self, pyfunc2, pyfunc3, func_name):
        m, n, k = 2, 3, 4

        def samples(m, n, k):
            for order_a, order_b in product('CF', 'CF'):
                a = self.sample_matrix(m, k, np.float64).copy(order=order_a)
                b = self.sample_matrix(k, n, np.float64).copy(order=order_b)
                yield a, b
            for dtype in self.dtypes:
                a = self.sample_matrix(m, k, dtype)
                b = self.sample_matrix(k, n, dtype)
                yield a, b

        cfunc2 = jit(nopython=True)(pyfunc2)
        for a, b in samples(m, n, k):
            self.check_func(pyfunc2, cfunc2, (a, b))
            self.check_func(pyfunc2, cfunc2, (b.T, a.T))
        if pyfunc3 is not None:
            cfunc3 = jit(nopython=True)(pyfunc3)
            for a, b in samples(m, n, k):
                out = np.empty((m, n), dtype=a.dtype)
                self.check_func_out(pyfunc3, cfunc3, (a, b), out)
                out = np.empty((n, m), dtype=a.dtype)
                self.check_func_out(pyfunc3, cfunc3, (b.T, a.T), out)

        # Mismatching sizes
        a = self.sample_matrix(m, k - 1, np.float64)
        b = self.sample_matrix(k, n, np.float64)
        self.assert_mismatching_sizes(cfunc2, (a, b))
        if pyfunc3 is not None:
            out = np.empty((m, n), np.float64)
            self.assert_mismatching_sizes(cfunc3, (a, b, out))
            a = self.sample_matrix(m, k, np.float64)
            b = self.sample_matrix(k, n, np.float64)
            out = np.empty((m, n - 1), np.float64)
            self.assert_mismatching_sizes(cfunc3, (a, b, out), is_out=True)
        # Mismatching dtypes
        a = self.sample_matrix(m, k, np.float32)
        b = self.sample_matrix(k, n, np.float64)
        self.assert_mismatching_dtypes(cfunc2, (a, b), func_name)
        if pyfunc3 is not None:
            a = self.sample_matrix(m, k, np.float64)
            b = self.sample_matrix(k, n, np.float64)
            out = np.empty((m, n), np.float32)
            self.assert_mismatching_dtypes(cfunc3, (a, b, out), func_name)

    def test_dot_mm(self):
        """
        Test matrix * matrix np.dot()
        """
        self.check_dot_mm(dot2, dot3, "np.dot()")

    @needs_matmul
    def test_matmul_vv(self):
        """
        Test vector @ vector
        """
        self.check_dot_vv(matmul_usecase, "'@'")

    @needs_matmul
    def test_matmul_vm(self):
        """
        Test vector @ matrix and matrix @ vector
        """
        self.check_dot_vm(matmul_usecase, None, "'@'")

    @needs_matmul
    def test_matmul_mm(self):
        """
        Test matrix @ matrix
        """
        self.check_dot_mm(matmul_usecase, None, "'@'")


if __name__ == '__main__':
    unittest.main()
