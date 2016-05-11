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


# Implementation definitions for the purpose of jitting.

def invert_matrix(a):
    return np.linalg.inv(a)


def cholesky_matrix(a):
    return np.linalg.cholesky(a)


def eig_matrix(a):
    return np.linalg.eig(a)


def svd_matrix(a, full_matrices=1):
    return np.linalg.svd(a, full_matrices)


def qr_matrix(a):
    return np.linalg.qr(a)


class TestLinalgBase(TestCase):
    """
    Provides setUp and common data/error modes for testing np.linalg functions.
    """

    def setUp(self):
        self.dtypes = (np.float64, np.float32, np.complex128, np.complex64)

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

    def orth_fact_sample_matrix(self, size, dtype, order):
        """
        Provides a sample matrix for use in orthogonal factorization tests.

        size: (rows, columns), the dimensions of the returned matrix.
        dtype: the dtype for the returned matrix.
        order: the memory layout for the returned matrix, 'F' or 'C'.
        """
        # May be worth just explicitly constructing this system at some point
        # with a kwarg for condition number?
        break_at = 10
        jmp = 0
        # have a few attempts at shuffling to get the condition number down
        # else not worry about it
        mn = size[0] * size[1]
        np.random.seed(0)  # repeatable seed
        while jmp < break_at:
            v = self.sample_vector(mn, dtype)
            # shuffle to improve conditioning
            np.random.shuffle(v)
            A = np.reshape(v, size)
            if np.linalg.cond(A) < mn:
                return np.array(A, order=order, dtype=dtype)
            jmp += 1
        return A

    def assert_error(self, cfunc, args, msg, err=ValueError):
        with self.assertRaises(err) as raises:
            cfunc(*args)
        self.assertIn(msg, str(raises.exception))

    def assert_non_square(self, cfunc, args):
        msg = "Last 2 dimensions of the array must be square."
        self.assert_error(cfunc, args, msg, np.linalg.LinAlgError)

    def assert_wrong_dtype(self, name, cfunc, args):
        msg = "np.linalg.%s() only supported on float and complex arrays" % name
        self.assert_error(cfunc, args, msg, errors.TypingError)

    def assert_wrong_dimensions(self, name, cfunc, args):
        msg = "np.linalg.%s() only supported on 2-D arrays" % name
        self.assert_error(cfunc, args, msg, errors.TypingError)

    def assert_no_nan_or_inf(self, cfunc, args):
        msg = "Array must not contain infs or NaNs."
        self.assert_error(cfunc, args, msg, np.linalg.LinAlgError)

    def assert_contig_sanity(self, got, expected_contig):
        """
        This checks that in a computed result from numba (array, possibly tuple
        of arrays) all the arrays are contiguous in memory and that they are
        all at least one of "C_CONTIGUOUS" or "F_CONTIGUOUS". The computed
        result of the contiguousness is then compared against a hardcoded
        expected result.

        got: is the computed results from numba
        expected_contig: is "C" or "F" and is the expected type of
                        contiguousness across all input values
                        (and therefore tests).
        """

        if isinstance(got, tuple):
            # tuple present, check all results
            c_contig = {a.flags.c_contiguous for a in got} == {True}
            f_contig = {a.flags.f_contiguous for a in got} == {True}
        else:
            # else a single array is present
            c_contig = got.flags.c_contiguous
            f_contig = got.flags.f_contiguous

        # check that the result (possible set of) is at least one of
        # C or F contiguous.
        msg = "Results are not at least one of all C or F contiguous."
        self.assertTrue(c_contig | f_contig, msg)

        msg = "Computed contiguousness does not match expected."
        if expected_contig == "C":
            self.assertTrue(c_contig, msg)
        elif expected_contig == "F":
            self.assertTrue(f_contig, msg)
        else:
            raise ValueError("Unknown contig")


class TestLinalgInv(TestLinalgBase):
    """
    Tests for np.linalg.inv.
    """

    def sample_matrix(self, m, dtype, order):
        a = np.zeros((m, m), dtype, order)
        a += np.diag(self.sample_vector(m, dtype))
        return a

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
                np.testing.assert_array_almost_equal_nulp(
                    got, expected, **kwargs)
                # check that the computed results are contig and in the same
                # way
                self.assert_contig_sanity(got, "C")
                del got, expected

        for dtype, order in product(self.dtypes, 'CF'):
            a = self.sample_matrix(n, dtype, order)
            check(a, nulp=3)

        for order in 'CF':
            a = np.array(((2, 1), (2, 3)), dtype=np.float64, order=order)
            check(a)

        # Non square matrices
        self.assert_non_square(cfunc, (np.ones((2, 3)),))

        # Wrong dtype
        self.assert_wrong_dtype("inv", cfunc,
                                (np.ones((2, 2), dtype=np.int32),))

        # Dimension issue
        self.assert_wrong_dimensions("inv", cfunc, (np.ones(10),))

        # Singular matrix
        self.assert_singular_matrix(cfunc, (np.zeros((2, 2)),))


class TestLinalgCholesky(TestLinalgBase):
    """
    Tests for np.linalg.cholesky.
    """

    def sample_matrix(self, m, dtype, order):
        # pd. (positive definite) matrix has eigenvalues in Z+
        np.random.seed(0)  # repeatable seed
        A = np.random.rand(m, m)
        # orthonormal q needed to form up q^{-1}*D*q
        # no "orth()" in numpy
        q, _ = np.linalg.qr(A)
        L = np.arange(1, m + 1)  # some positive eigenvalues
        Q = np.dot(np.dot(q.T, np.diag(L)), q)  # construct
        Q = np.array(Q, dtype=dtype, order=order)  # sort out order/type
        return Q

    def assert_not_pd(self, cfunc, args):
        msg = "Matrix is not positive definite."
        self.assert_error(cfunc, args, msg, np.linalg.LinAlgError)

    @needs_lapack
    def test_linalg_cholesky(self):
        """
        Test np.linalg.cholesky
        """
        n = 10
        cfunc = jit(nopython=True)(cholesky_matrix)

        def check(a):
            expected = cholesky_matrix(a)
            got = cfunc(a)
            use_reconstruction = False
            # check that the computed results are contig and in the same way
            self.assert_contig_sanity(got, "C")

            # try strict
            try:
                np.testing.assert_array_almost_equal_nulp(got, expected,
                                                          nulp=10)
            except AssertionError:
                # fall back to reconstruction
                use_reconstruction = True

            # try via reconstruction
            if use_reconstruction:
                rec = np.dot(got, np.conj(got.T))
                resolution = 5 * np.finfo(a.dtype).resolution
                np.testing.assert_allclose(
                    a,
                    rec,
                    rtol=resolution,
                    atol=resolution
                )

            # Ensure proper resource management
            with self.assertNoNRTLeak():
                cfunc(a)

        for dtype, order in product(self.dtypes, 'FC'):
            a = self.sample_matrix(n, dtype, order)
            check(a)

        rn = "cholesky"
        # Non square matrices
        self.assert_non_square(cfunc, (np.ones((2, 3), dtype=np.float64),))

        # Wrong dtype
        self.assert_wrong_dtype(rn, cfunc,
                                (np.ones((2, 2), dtype=np.int32),))

        # Dimension issue
        self.assert_wrong_dimensions(rn, cfunc,
                                     (np.ones(10, dtype=np.float64),))

        # not pd
        self.assert_not_pd(cfunc,
                           (np.ones(4, dtype=np.float64).reshape(2, 2),))


class TestLinalgEig(TestLinalgBase):
    """
    Tests for np.linalg.eig.
    """

    def sample_matrix(self, m, dtype, order):
        # This is a tridiag with the same but skewed values on the diagonals
        v = self.sample_vector(m, dtype)
        Q = np.diag(v)
        idx = np.nonzero(np.eye(Q.shape[0], Q.shape[1], 1))
        Q[idx] = v[1:]
        idx = np.nonzero(np.eye(Q.shape[0], Q.shape[1], -1))
        Q[idx] = v[:-1]
        Q = np.array(Q, dtype=dtype, order=order)
        return Q

    def assert_no_domain_change(self, cfunc, args):
        msg = "eig() argument must not cause a domain change."
        self.assert_error(cfunc, args, msg)

    @needs_lapack
    def test_linalg_eig(self):
        """
        Test np.linalg.eig
        """
        n = 10
        cfunc = jit(nopython=True)(eig_matrix)

        def check(a):
            expected = eig_matrix(a)
            got = cfunc(a)
            # check that the returned tuple is same length
            self.assertEqual(len(expected), len(got))
            # and that length is 2
            self.assertEqual(len(got), 2)
            # and that the computed results are contig and in the same way
            self.assert_contig_sanity(got, "F")

            use_reconstruction = False
            # try plain match of each array to np first
            for k in range(len(expected)):
                try:
                    np.testing.assert_array_almost_equal_nulp(
                        got[k], expected[k], nulp=10)
                except AssertionError:
                    # plain match failed, test by reconstruction
                    use_reconstruction = True

            # if plain match fails then reconstruction is used.
            # this checks that A*V ~== V*W
            # i.e. eigensystem ties out
            # this is required as numpy uses only double precision lapack
            # routines and computation of eigenvectors is numerically
            # sensitive, numba using the type specific routines therefore
            # sometimes comes out with a different (but entirely
            # valid) answer (eigenvectors are not unique etc.).
            if use_reconstruction:
                w, v = got
                lhs = np.dot(a, v)
                rhs = np.dot(v, np.diag(w))
                resolution = 5 * np.finfo(a.dtype).resolution
                np.testing.assert_allclose(
                    lhs,
                    rhs,
                    rtol=resolution,
                    atol=resolution
                )

            # Ensure proper resource management
            with self.assertNoNRTLeak():
                cfunc(a)

        for dtype, order in product(self.dtypes, 'FC'):
            a = self.sample_matrix(n, dtype, order)
            check(a)

        rn = "eig"

        # test both a real and complex type as the impls are different
        for ty in [np.float32, np.complex64]:
            # Non square matrices
            self.assert_non_square(cfunc, (np.ones((2, 3), dtype=ty),))

            # Wrong dtype
            self.assert_wrong_dtype(rn, cfunc,
                                    (np.ones((2, 2), dtype=np.int32),))

            # Dimension issue
            self.assert_wrong_dimensions(rn, cfunc, (np.ones(10, dtype=ty),))

            # no nans or infs
            self.assert_no_nan_or_inf(cfunc,
                                      (np.array([[1., 2., ], [np.inf, np.nan]],
                                                dtype=ty),))

        # By design numba does not support dynamic return types, numpy does
        # and uses this in the case of returning eigenvalues/vectors of
        # a real matrix. The return type of np.linalg.eig(), when
        # operating on a matrix in real space depends on the values present
        # in the matrix itself (recalling that eigenvalues are the roots of the
        # characteristic polynomial of the system matrix, which will by
        # construction depend on the values present in the system matrix).
        # This test asserts that if a domain change is required on the return
        # type, i.e. complex eigenvalues from a real input, an error is raised.
        # For complex types, regardless of the value of the imaginary part of
        # the returned eigenvalues, a complex type will be returned, this
        # follows numpy and fits in with numba.

        # First check that the computation is valid (i.e. in complex space)
        A = np.array([[1, -2], [2, 1]])
        check(A.astype(np.complex128))
        # and that the imaginary part is nonzero
        l, _ = eig_matrix(A)
        self.assertTrue(np.any(l.imag))

        # Now check that the computation fails in real space
        for ty in [np.float32, np.float64]:
            self.assert_no_domain_change(cfunc, (A.astype(ty),))


class TestLinalgSvd(TestLinalgBase):
    """
    Tests for np.linalg.svd.
    """

    @needs_lapack
    def test_linalg_svd(self):
        """
        Test np.linalg.svd
        """
        cfunc = jit(nopython=True)(svd_matrix)

        def check(a, **kwargs):
            expected = svd_matrix(a, **kwargs)
            got = cfunc(a, **kwargs)
            # check that the returned tuple is same length
            self.assertEqual(len(expected), len(got))
            # and that length is 3
            self.assertEqual(len(got), 3)
            # and that the computed results are contig and in the same way
            self.assert_contig_sanity(got, "F")

            use_reconstruction = False
            # try plain match of each array to np first
            for k in range(len(expected)):

                try:
                    np.testing.assert_array_almost_equal_nulp(
                        got[k], expected[k], nulp=10)
                except AssertionError:
                    # plain match failed, test by reconstruction
                    use_reconstruction = True

            # if plain match fails then reconstruction is used.
            # this checks that A ~= U*S*V**H
            # i.e. SV decomposition ties out
            # this is required as numpy uses only double precision lapack
            # routines and computation of svd is numerically
            # sensitive, numba using the type specific routines therefore
            # sometimes comes out with a different answer (orthonormal bases
            # are not unique etc.).
            if use_reconstruction:
                u, sv, vt = got

                # check they are dimensionally correct
                for k in range(len(expected)):
                    self.assertEqual(got[k].shape, expected[k].shape)

                # regardless of full_matrices cols in u and rows in vt
                # dictates the working size of s
                s = np.zeros((u.shape[1], vt.shape[0]))
                np.fill_diagonal(s, sv)

                rec = np.dot(np.dot(u, s), vt)
                resolution = np.finfo(a.dtype).resolution
                np.testing.assert_allclose(
                    a,
                    rec,
                    rtol=10 * resolution,
                    atol=100 * resolution  # zeros tend to be fuzzy
                )

            # Ensure proper resource management
            with self.assertNoNRTLeak():
                cfunc(a, **kwargs)

        # test: column vector, tall, wide, square, row vector
        # prime sizes
        sizes = [(7, 1), (7, 5), (5, 7), (3, 3), (1, 7)]

        # flip on reduced or full matrices
        full_matrices = (True, False)

        # test loop
        for size, dtype, fmat, order in \
                product(sizes, self.dtypes, full_matrices, 'FC'):

            a = self.orth_fact_sample_matrix(size, dtype, order)
            check(a, full_matrices=fmat)

        rn = "svd"

        # Wrong dtype
        self.assert_wrong_dtype(rn, cfunc,
                                (np.ones((2, 2), dtype=np.int32),))

        # Dimension issue
        self.assert_wrong_dimensions(rn, cfunc,
                                     (np.ones(10, dtype=np.float64),))

        # no nans or infs
        self.assert_no_nan_or_inf(cfunc,
                                  (np.array([[1., 2., ], [np.inf, np.nan]],
                                            dtype=np.float64),))


class TestLinalgQr(TestLinalgBase):
    """
    Tests for np.linalg.qr.
    """
    
    @needs_lapack
    def test_linalg_qr(self):
        """
        Test np.linalg.qr
        """
        cfunc = jit(nopython=True)(qr_matrix)

        def check(a, **kwargs):
            expected = qr_matrix(a, **kwargs)
            got = cfunc(a, **kwargs)

            # check that the returned tuple is same length
            self.assertEqual(len(expected), len(got))
            # and that length is 2
            self.assertEqual(len(got), 2)
            # and that the computed results are contig and in the same way
            self.assert_contig_sanity(got, "F")

            use_reconstruction = False
            # try plain match of each array to np first
            for k in range(len(expected)):
                try:
                    np.testing.assert_array_almost_equal_nulp(
                        got[k], expected[k], nulp=10)
                except AssertionError:
                    # plain match failed, test by reconstruction
                    use_reconstruction = True

            # if plain match fails then reconstruction is used.
            # this checks that A ~= Q*R and that (Q^H)*Q = I
            # i.e. QR decomposition ties out
            # this is required as numpy uses only double precision lapack
            # routines and computation of qr is numerically
            # sensitive, numba using the type specific routines therefore
            # sometimes comes out with a different answer (orthonormal bases
            # are not unique etc.).
            if use_reconstruction:
                q, r = got

                # check they are dimensionally correct
                for k in range(len(expected)):
                    self.assertEqual(got[k].shape, expected[k].shape)

                # check A=q*r
                rec = np.dot(q, r)
                resolution = np.finfo(a.dtype).resolution
                np.testing.assert_allclose(
                    a,
                    rec,
                    rtol=10 * resolution,
                    atol=100 * resolution  # zeros tend to be fuzzy
                )

                # check q is orthonormal
                np.testing.assert_allclose(
                    np.eye(min(a.shape), dtype=a.dtype),
                    np.dot(np.conjugate(q.T), q),
                    rtol=resolution,
                    atol=resolution
                )

            # Ensure proper resource management
            with self.assertNoNRTLeak():
                cfunc(a, **kwargs)

        # test: column vector, tall, wide, square, row vector
        # prime sizes
        sizes = [(7, 1), (11, 5), (5, 11), (3, 3), (1, 7)]

        # test loop
        for size, dtype, order in \
                product(sizes, self.dtypes, 'FC'):
            a = self.orth_fact_sample_matrix(size, dtype, order)
            check(a)

        rn = "qr"

        # Wrong dtype
        self.assert_wrong_dtype(rn, cfunc,
                                (np.ones((2, 2), dtype=np.int32),))

        # Dimension issue
        self.assert_wrong_dimensions(rn, cfunc,
                                     (np.ones(10, dtype=np.float64),))

        # no nans or infs
        self.assert_no_nan_or_inf(cfunc,
                                  (np.array([[1., 2., ], [np.inf, np.nan]],
                                            dtype=np.float64),))

if __name__ == '__main__':
    unittest.main()
