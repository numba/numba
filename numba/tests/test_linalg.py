from __future__ import division, print_function

import contextlib
import gc
from itertools import product
import sys
import warnings

import numpy as np

from numba import unittest_support as unittest
from numba import jit, errors
from .support import TestCase, tag
from .matmul_usecase import matmul_usecase, needs_matmul, needs_blas

try:
    import scipy.linalg.cython_lapack
    has_lapack = True
except ImportError:
    has_lapack = False

needs_lapack = unittest.skipUnless(has_lapack,
                                   "LAPACK needs Scipy 0.16+")

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

    def setUp(self):
        # Collect leftovers from previous test cases before checking for leaks
        gc.collect()

    def sample_vector(self, n, dtype):
        # Be careful to generate only exactly representable float values,
        # to avoid rounding discrepancies between Numpy and Numba
        base = np.arange(n)
        if issubclass(dtype, np.complexfloating):
            return (base * (1 - 0.5j) + 2j).astype(dtype)
        else:
            return (base * 0.5 - 1).astype(dtype)

    def sample_matrix(self, m, n, dtype):
        return self.sample_vector(m * n, dtype).reshape((m, n))

    @contextlib.contextmanager
    def check_contiguity_warning(self, pyfunc):
        """
        Check performance warning(s) for non-contiguity.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', errors.PerformanceWarning)
            yield
        self.assertGreaterEqual(len(w), 1)
        self.assertIs(w[0].category, errors.PerformanceWarning)
        self.assertIn("faster on contiguous arrays", str(w[0].message))
        self.assertEqual(w[0].filename, pyfunc.__code__.co_filename)
        # This works because our functions are one-liners
        self.assertEqual(w[0].lineno, pyfunc.__code__.co_firstlineno + 1)

    def check_func(self, pyfunc, cfunc, args):
        with self.assertNoNRTLeak():
            expected = pyfunc(*args)
            got = cfunc(*args)
            self.assertPreciseEqual(got, expected, ignore_sign_on_zero=True)
            del got, expected

    def check_func_out(self, pyfunc, cfunc, args, out):
        with self.assertNoNRTLeak():
            expected = np.copy(out)
            got = np.copy(out)
            self.assertIs(pyfunc(*args, out=expected), expected)
            self.assertIs(cfunc(*args, out=got), got)
            self.assertPreciseEqual(got, expected, ignore_sign_on_zero=True)
            del got, expected

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
            # Non-contiguous
            self.check_func(pyfunc, cfunc, (a[::-1], b[::-1]))

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
            # Non-contiguous
            yield a[::-1], b[::-1]

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

        def samples(m, n, k):
            for order_a, order_b in product('CF', 'CF'):
                a = self.sample_matrix(m, k, np.float64).copy(order=order_a)
                b = self.sample_matrix(k, n, np.float64).copy(order=order_b)
                yield a, b
            for dtype in self.dtypes:
                a = self.sample_matrix(m, k, dtype)
                b = self.sample_matrix(k, n, dtype)
                yield a, b
            # Non-contiguous
            yield a[::-1], b[::-1]

        cfunc2 = jit(nopython=True)(pyfunc2)
        if pyfunc3 is not None:
            cfunc3 = jit(nopython=True)(pyfunc3)

        # Test generic matrix * matrix as well as "degenerate" cases
        # where one of the outer dimensions is 1 (i.e. really represents
        # a vector, which may select a different implementation)
        for m, n, k in [(2, 3, 4),  # Generic matrix * matrix
                        (1, 3, 4),  # 2d vector * matrix
                        (1, 1, 4),  # 2d vector * 2d vector
                        ]:
            for a, b in samples(m, n, k):
                self.check_func(pyfunc2, cfunc2, (a, b))
                self.check_func(pyfunc2, cfunc2, (b.T, a.T))
            if pyfunc3 is not None:
                for a, b in samples(m, n, k):
                    out = np.empty((m, n), dtype=a.dtype)
                    self.check_func_out(pyfunc3, cfunc3, (a, b), out)
                    out = np.empty((n, m), dtype=a.dtype)
                    self.check_func_out(pyfunc3, cfunc3, (b.T, a.T), out)

        # Mismatching sizes
        m, n, k = 2, 3, 4
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

    @tag('important')
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

    @needs_blas
    def test_contiguity_warnings(self):
        m, k, n = 2, 3, 4
        dtype = np.float64
        a = self.sample_matrix(m, k, dtype)[::-1]
        b = self.sample_matrix(k, n, dtype)[::-1]
        out = np.empty((m, n), dtype)

        cfunc = jit(nopython=True)(dot2)
        with self.check_contiguity_warning(cfunc.py_func):
            cfunc(a, b)
        cfunc = jit(nopython=True)(dot3)
        with self.check_contiguity_warning(cfunc.py_func):
            cfunc(a, b, out)

        a = self.sample_vector(n, dtype)[::-1]
        b = self.sample_vector(n, dtype)[::-1]

        cfunc = jit(nopython=True)(vdot)
        with self.check_contiguity_warning(cfunc.py_func):
            cfunc(a, b)


def invert_matrix(a):
    return np.linalg.inv(a)


class TestLinalgInv(TestCase):
    """
    Tests for np.linalg.inv.
    """

    dtypes = (np.float64, np.float32, np.complex128, np.complex64)

    def setUp(self):
        # Collect leftovers from previous test cases before checking for leaks
        gc.collect()

    def sample_vector(self, n, dtype):
        # Be careful to generate only exactly representable float values,
        # to avoid rounding discrepancies between Numpy and Numba
        base = np.arange(n)
        if issubclass(dtype, np.complexfloating):
            return (base * (1 - 0.5j) + 2j).astype(dtype)
        else:
            return (base * 0.5 + 1).astype(dtype)

    def sample_matrix(self, m, dtype, order):
        a = np.zeros((m, m), dtype, order)
        a += np.diag(self.sample_vector(m, dtype))
        return a

    def assert_error(self, cfunc, args, msg, err=ValueError):
        with self.assertRaises(err) as raises:
            cfunc(*args)
        self.assertIn(msg, str(raises.exception))

    def assert_non_square(self, cfunc, args):
        msg = "np.linalg.inv can only work on square arrays."
        self.assert_error(cfunc, args, msg)

    def assert_wrong_dtype(self, cfunc, args):
        msg = "np.linalg.inv() only supported on float and complex arrays"
        self.assert_error(cfunc, args, msg, errors.TypingError)

    def assert_wrong_dimensions(self, cfunc, args):
        msg = "np.linalg.inv() only supported on 2-D arrays"
        self.assert_error(cfunc, args, msg, errors.TypingError)

    def assert_singular_matrix(self, cfunc, args):
        msg = "Matrix is singular and cannot be inverted"
        self.assert_error(cfunc, args, msg)

    @tag('important')
    @needs_lapack
    def test_linalg_inv(self):
        """
        Test np.linalg.inv
        """
        n = 10
        cfunc = jit(nopython=True)(invert_matrix)

        def check(a, **kwargs):
            with self.assertNoNRTLeak():
                expected = invert_matrix(a).copy(order='C')
                got = cfunc(a)
                # XXX had to use that function otherwise comparison fails
                # because of +0, -0 discrepancies
                np.testing.assert_array_almost_equal_nulp(got, expected, **kwargs)
                del got, expected

        for dtype, order in product(self.dtypes, 'CF'):
            a = self.sample_matrix(n, dtype, order)
            check(a, nulp=3)

        for order in 'CF':
            a = np.array(((2, 1), (2, 3)), dtype=np.float64, order=order)
            check(a)

        # Non square matrices
        self.assert_non_square(cfunc, (np.ones((2,3)),))

        # Wrong dtype
        self.assert_wrong_dtype(cfunc,
                                (np.ones((2, 2), dtype=np.int32),))

        # Dimension issue
        self.assert_wrong_dimensions(cfunc, (np.ones(10),))

        # Singular matrix
        self.assert_singular_matrix(cfunc, (np.zeros((2, 2)),))


if __name__ == '__main__':
    unittest.main()
