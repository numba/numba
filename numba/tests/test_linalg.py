from __future__ import division, print_function

import contextlib
import gc
from itertools import product, cycle
import sys
import warnings
from numbers import Number, Integral

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


def lstsq_system(A, B, rcond=-1):
    return np.linalg.lstsq(A, B, rcond)


def solve_system(A, B):
    return np.linalg.solve(A, B)


def pinv_matrix(A, rcond=1e-15):  # 1e-15 from numpy impl
    return np.linalg.pinv(A)


class TestLinalgBase(TestCase):
    """
    Provides setUp and common data/error modes for testing np.linalg functions.
    """

    # supported dtypes
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

    def specific_sample_matrix(
            self, size, dtype, order, rank=None, condition=None):
        """
        Provides a sample matrix with an optionally specified rank or condition
        number.

        size: (rows, columns), the dimensions of the returned matrix.
        dtype: the dtype for the returned matrix.
        order: the memory layout for the returned matrix, 'F' or 'C'.
        rank: the rank of the matrix, an integer value, defaults to full rank.
        condition: the condition number of the matrix (defaults to 1.)

        NOTE: Only one of rank or condition may be set.
        """

        # default condition
        d_cond = 1.

        if len(size) != 2:
            raise ValueError("size must be a length 2 tuple.")

        if order not in ['F', 'C']:
            raise ValueError("order must be one of 'F' or 'C'.")

        if dtype not in [np.float32, np.float64, np.complex64, np.complex128]:
            raise ValueError("dtype must be a numpy floating point type.")

        if rank is not None and condition is not None:
            raise ValueError("Only one of rank or condition can be specified.")

        if condition is None:
            condition = d_cond

        if condition < 1:
            raise ValueError("Condition number must be >=1.")

        np.random.seed(0)  # repeatable seed
        m, n = size

        if m < 0 or n < 0:
            raise ValueError("Negative dimensions given for matrix shape.")

        minmn = min(m, n)
        if rank is None:
            rv = minmn
        else:
            if rank <= 0:
                raise ValueError("Rank must be greater than zero.")
            if not isinstance(rank, Integral):
                raise ValueError("Rank must an integer.")
            rv = rank
            if rank > minmn:
                raise ValueError("Rank given greater than full rank.")

        if m == 1 or n == 1:
            # vector, must be rank 1 (enforced above)
            # condition of vector is also 1
            if condition != d_cond:
                raise ValueError(
                    "Condition number was specified for a vector (always 1.).")
            maxmn = max(m, n)
            Q = self.sample_vector(maxmn, dtype).reshape(m, n)
        else:
            # Build a sample matrix via combining SVD like inputs.

            # Create matrices of left and right singular vectors.
            # This could use Modified Gram-Schmidt and perhaps be quicker,
            # at present it uses QR decompositions to obtain orthonormal
            # matrices.
            tmp = self.sample_vector(m * m, dtype).reshape(m, m)
            U, _ = np.linalg.qr(tmp)
            tmp = self.sample_vector(n * n, dtype).reshape(n, n)
            V, _ = np.linalg.qr(tmp)
            # create singular values.
            sv = np.linspace(d_cond, condition, rv)
            S = np.zeros((m, n))
            idx = np.nonzero(np.eye(m, n))
            S[idx[0][:rv], idx[1][:rv]] = sv
            Q = np.dot(np.dot(U, S), V.T)  # construct
            Q = np.array(Q, dtype=dtype, order=order)  # sort out order/type

        return Q

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
            for a in got:
                self.assert_contig_sanity(a, expected_contig)
        else:
            if not isinstance(got, Number):
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

    def assert_raise_on_singular(self, cfunc, args):
        msg = "Matrix is singular to machine precision."
        self.assert_error(cfunc, args, msg, err=np.linalg.LinAlgError)

    def assert_is_identity_matrix(self, got, rtol=None, atol=None):
        """
        Checks if a matrix is equal to the identity matrix.
        """
        # check it is square
        self.assertEqual(got.shape[-1], got.shape[-2])
        # create identity matrix
        eye = np.eye(got.shape[-1], dtype=got.dtype)
        resolution = 5 * np.finfo(got.dtype).resolution
        if rtol is None:
            rtol = 10 * resolution
        if atol is None:
            atol = 100 * resolution  # zeros tend to be fuzzy
        # check it matches
        np.testing.assert_allclose(got, eye, rtol, atol)


class TestTestLinalgBase(TestCase):
    """
    The sample matrix code TestLinalgBase.specific_sample_matrix()
    is a bit involved, this class tests it works as intended.
    """

    def test_specific_sample_matrix(self):

        # add a default test to the ctor, it never runs so doesn't matter
        inst = TestLinalgBase('specific_sample_matrix')

        sizes = [(7, 1), (11, 5), (5, 11), (3, 3), (1, 7)]

        # test loop
        for size, dtype, order in product(sizes, inst.dtypes, 'FC'):

            m, n = size
            minmn = min(m, n)

            # test default full rank
            A = inst.specific_sample_matrix(size, dtype, order)
            self.assertEqual(A.shape, size)
            self.assertEqual(np.linalg.matrix_rank(A), minmn)

            # test reduced rank if a reduction is possible
            if minmn > 1:
                rank = minmn - 1
                A = inst.specific_sample_matrix(size, dtype, order, rank=rank)
                self.assertEqual(A.shape, size)
                self.assertEqual(np.linalg.matrix_rank(A), rank)

            resolution = 5 * np.finfo(dtype).resolution

            # test default condition
            A = inst.specific_sample_matrix(size, dtype, order)
            self.assertEqual(A.shape, size)
            np.testing.assert_allclose(np.linalg.cond(A),
                                       1.,
                                       rtol=resolution,
                                       atol=resolution)

            # test specified condition if matrix is > 1D
            if minmn > 1:
                condition = 10.
                A = inst.specific_sample_matrix(
                    size, dtype, order, condition=condition)
                self.assertEqual(A.shape, size)
                np.testing.assert_allclose(np.linalg.cond(A),
                                           10.,
                                           rtol=resolution,
                                           atol=resolution)

        # check errors are raised appropriately
        def check_error(args, msg, err=ValueError):
            with self.assertRaises(err) as raises:
                inst.specific_sample_matrix(*args)
            self.assertIn(msg, str(raises.exception))

        # check the checker runs ok
        with self.assertRaises(AssertionError) as raises:
            msg = "blank"
            check_error(((2, 3), np.float64, 'F'), msg, err=ValueError)

        # check invalid inputs...

        # bad size
        msg = "size must be a length 2 tuple."
        check_error(((1,), np.float64, 'F'), msg, err=ValueError)

        # bad order
        msg = "order must be one of 'F' or 'C'."
        check_error(((2, 3), np.float64, 'z'), msg, err=ValueError)

        # bad type
        msg = "dtype must be a numpy floating point type."
        check_error(((2, 3), np.int32, 'F'), msg, err=ValueError)

        # specifying both rank and condition
        msg = "Only one of rank or condition can be specified."
        check_error(((2, 3), np.float64, 'F', 1, 1), msg, err=ValueError)

        # specifying negative condition
        msg = "Condition number must be >=1."
        check_error(((2, 3), np.float64, 'F', None, -1), msg, err=ValueError)

        # specifying negative matrix dimension
        msg = "Negative dimensions given for matrix shape."
        check_error(((2, -3), np.float64, 'F'), msg, err=ValueError)

        # specifying negative rank
        msg = "Rank must be greater than zero."
        check_error(((2, 3), np.float64, 'F', -1), msg, err=ValueError)

        # specifying a rank greater than maximum rank
        msg = "Rank given greater than full rank."
        check_error(((2, 3), np.float64, 'F', 4), msg, err=ValueError)

        # specifying a condition number for a vector
        msg = "Condition number was specified for a vector (always 1.)."
        check_error(((1, 3), np.float64, 'F', None, 10), msg, err=ValueError)

        # specifying a non integer rank
        msg = "Rank must an integer."
        check_error(((2, 3), np.float64, 'F', 1.5), msg, err=ValueError)


class TestLinalgInv(TestLinalgBase):
    """
    Tests for np.linalg.inv.
    """

    @tag('important')
    @needs_lapack
    def test_linalg_inv(self):
        """
        Test np.linalg.inv
        """
        n = 10
        cfunc = jit(nopython=True)(invert_matrix)

        def check(a, **kwargs):
            expected = invert_matrix(a)
            got = cfunc(a)
            self.assert_contig_sanity(got, "F")

            use_reconstruction = False

            # try strict
            try:
                np.testing.assert_array_almost_equal_nulp(got, expected,
                                                          nulp=10)
            except AssertionError:
                # fall back to reconstruction
                use_reconstruction = True

            if use_reconstruction:
                rec = np.dot(got, a)
                self.assert_is_identity_matrix(rec)

            # Ensure proper resource management
            with self.assertNoNRTLeak():
                cfunc(a)

        for dtype, order in product(self.dtypes, 'CF'):
            a = self.specific_sample_matrix((n, n), dtype, order)
            check(a)

        # Non square matrix
        self.assert_non_square(cfunc, (np.ones((2, 3)),))

        # Wrong dtype
        self.assert_wrong_dtype("inv", cfunc,
                                (np.ones((2, 2), dtype=np.int32),))

        # Dimension issue
        self.assert_wrong_dimensions("inv", cfunc, (np.ones(10),))

        # Singular matrix
        self.assert_raise_on_singular(cfunc, (np.zeros((2, 2)),))


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

            a = self.specific_sample_matrix(size, dtype, order)
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
                self.assert_is_identity_matrix(np.dot(np.conjugate(q.T), q))

            # Ensure proper resource management
            with self.assertNoNRTLeak():
                cfunc(a, **kwargs)

        # test: column vector, tall, wide, square, row vector
        # prime sizes
        sizes = [(7, 1), (11, 5), (5, 11), (3, 3), (1, 7)]

        # test loop
        for size, dtype, order in \
                product(sizes, self.dtypes, 'FC'):
            a = self.specific_sample_matrix(size, dtype, order)
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


class TestLinalgSystems(TestLinalgBase):
    """
    Base class for testing "system" solvers from np.linalg.
    Namely np.linalg.solve() and np.linalg.lstsq().
    """

    # check for RHS with dimension > 2 raises
    def assert_wrong_dimensions_1D(self, name, cfunc, args):
        msg = "np.linalg.%s() only supported on 1 and 2-D arrays" % name
        self.assert_error(cfunc, args, msg, errors.TypingError)

    # check that a dimensionally invalid system raises
    def assert_dimensionally_invalid(self, cfunc, args):
        msg = "Incompatible array sizes, system is not dimensionally valid."
        self.assert_error(cfunc, args, msg, np.linalg.LinAlgError)


class TestLinalgLstsq(TestLinalgSystems):
    """
    Tests for np.linalg.lstsq.
    """

    # NOTE: The testing of this routine is hard as it has to handle numpy
    # using double precision routines on single precision input, this has
    # a knock on effect especially in rank deficient cases and cases where
    # conditioning is generally poor. As a result computed ranks can differ
    # and consequently the calculated residual can differ.
    # The tests try and deal with this as best as they can through the use
    # of reconstruction and measures like residual norms.
    # Suggestions for improvements are welcomed!

    @needs_lapack
    def test_linalg_lstsq(self):
        """
        Test np.linalg.lstsq
        """
        cfunc = jit(nopython=True)(lstsq_system)

        def check(A, B, **kwargs):
            expected = lstsq_system(A, B, **kwargs)
            got = cfunc(A, B, **kwargs)

            # check that the returned tuple is same length
            self.assertEqual(len(expected), len(got))
            # and that length is 4
            self.assertEqual(len(got), 4)
            # and that the computed results are contig and in the same way
            self.assert_contig_sanity(got, "C")

            use_reconstruction = False

            # check the ranks are the same and continue to a standard
            # match if that is the case (if ranks differ, then output
            # in e.g. residual array is of different size!).
            try:
                self.assertEqual(got[2], expected[2])
                # try plain match of each array to np first
                for k in range(len(expected)):
                    try:
                        np.testing.assert_array_almost_equal_nulp(
                            got[k], expected[k], nulp=10)
                    except AssertionError:
                        # plain match failed, test by reconstruction
                        use_reconstruction = True
            except AssertionError:
                use_reconstruction = True

            if use_reconstruction:
                x, res, rank, s = got

                # indicies in the output which are ndarrays
                out_array_idx = [0, 1, 3]

                try:
                    # check the ranks are the same
                    self.assertEqual(rank, expected[2])
                    # check they are dimensionally correct, skip [2] = rank.
                    for k in out_array_idx:
                        if isinstance(expected[k], np.ndarray):
                            self.assertEqual(got[k].shape, expected[k].shape)
                except AssertionError:
                    # check the rank differs by 1. (numerical fuzz)
                    self.assertTrue(abs(rank - expected[2]) < 2)

                # check if A*X = B
                resolution = np.finfo(A.dtype).resolution
                try:
                    # this will work so long as the conditioning is
                    # ok and the rank is full
                    rec = np.dot(A, x)
                    np.testing.assert_allclose(
                        B,
                        rec,
                        rtol=10 * resolution,
                        atol=10 * resolution
                    )
                except AssertionError:
                    # system is probably under/over determined and/or
                    # poorly conditioned. Check slackened equality
                    # and that the residual norm is the same.
                    for k in out_array_idx:
                        try:
                            np.testing.assert_allclose(
                                expected[k],
                                got[k],
                                rtol=100 * resolution,
                                atol=100 * resolution
                            )
                        except AssertionError:
                            # check the fail is likely due to bad conditioning
                            c = np.linalg.cond(A)
                            self.assertGreater(10 * c, (1. / resolution))

                        # make sure the residual 2-norm is ok
                        # if this fails its probably due to numpy using double
                        # precision LAPACK routines for singles.
                        res_expected = np.linalg.norm(
                            B - np.dot(A, expected[0]))
                        res_got = np.linalg.norm(B - np.dot(A, x))
                        # rtol = 10. as all the systems are products of orthonormals
                        # and on the small side (rows, cols) < 100.
                        np.testing.assert_allclose(
                            res_expected, res_got, rtol=10.)

            # Ensure proper resource management
            with self.assertNoNRTLeak():
                cfunc(A, B, **kwargs)

        # test: column vector, tall, wide, square, row vector
        # prime sizes, the A's
        sizes = [(7, 1), (11, 5), (5, 11), (3, 3), (1, 7)]
        # compatible B's for Ax=B must have same number of rows and 1 or more
        # columns

        # This test takes ages! So combinations are trimmed via cycling

        # gets a dtype
        cycle_dt = cycle(self.dtypes)

        orders = ['F', 'C']
        # gets a memory order flag
        cycle_order = cycle(orders)

        # a specific condition number to use in the following tests
        # there is nothing special about it other than it is not magic
        specific_cond = 10.

        # inner test loop, extracted as there's additional logic etc required
        # that'd end up with this being repeated a lot
        def inner_test_loop_fn(A, dt, **kwargs):
            # test solve Ax=B for (column, matrix) B, same dtype as A
            b_sizes = (1, 13)

            for b_size in b_sizes:

                # check 2D B
                b_order = next(cycle_order)
                B = self.specific_sample_matrix(
                    (A.shape[0], b_size), dt, b_order)
                check(A, B, **kwargs)

                # check 1D B
                b_order = next(cycle_order)
                tmp = B[:, 0].copy(order=b_order)
                check(A, tmp, **kwargs)

        # test loop
        for a_size in sizes:

            # order and dtype
            a_dtype = next(cycle_dt)
            a_order = next(cycle_order)

            # A full rank, well conditioned system
            A = self.specific_sample_matrix(a_size, a_dtype, a_order)

            # run the test loop
            inner_test_loop_fn(A, a_dtype)

            m, n = a_size
            minmn = min(m, n)

            # operations that only make sense with a 2D matrix system
            if m != 1 and n != 1:

                # Test a rank deficient system
                r = minmn - 1
                # order and dtype
                a_dtype = next(cycle_dt)
                a_order = next(cycle_order)
                A = self.specific_sample_matrix(
                    a_size, a_dtype, a_order, rank=r)
                # run the test loop
                inner_test_loop_fn(A, a_dtype)

                # Test a system with a given condition number for use in
                # testing the rcond parameter.
                # This works because the singular values in the
                # specific_sample_matrix code are linspace (1, cond, [0... if
                # rank deficient])
                a_dtype = next(cycle_dt)
                a_order = next(cycle_order)
                A = self.specific_sample_matrix(
                    a_size, a_dtype, a_order, condition=specific_cond)
                # run the test loop
                rcond = 1. / specific_cond
                approx_half_rank_rcond = minmn * rcond
                inner_test_loop_fn(A, a_dtype,
                                   rcond=approx_half_rank_rcond)

        # Test input validation
        ok = np.array([[1., 2.], [3., 4.]], dtype=np.float64)

        # check ok input is ok
        cfunc, (ok, ok)

        # check bad inputs
        rn = "lstsq"

        # Wrong dtype
        bad = np.array([[1, 2], [3, 4]], dtype=np.int32)
        self.assert_wrong_dtype(rn, cfunc, (ok, bad))
        self.assert_wrong_dtype(rn, cfunc, (bad, ok))

        # Dimension issue
        bad = np.array([1, 2], dtype=np.float64)
        self.assert_wrong_dimensions(rn, cfunc, (bad, ok))

        # no nans or infs
        bad = np.array([[1., 2., ], [np.inf, np.nan]], dtype=np.float64)
        self.assert_no_nan_or_inf(cfunc, (ok, bad))
        self.assert_no_nan_or_inf(cfunc, (bad, ok))

        # check 1D is accepted for B (2D is done previously)
        # and then that anything of higher dimension raises
        oneD = np.array([1., 2.], dtype=np.float64)
        cfunc, (ok, oneD)
        bad = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
        self.assert_wrong_dimensions_1D(rn, cfunc, (ok, bad))

        # check a dimensionally invalid system raises (1D and 2D cases
        # checked)
        bad1D = np.array([1.], dtype=np.float64)
        bad2D = np.array([[1.], [2.], [3.]], dtype=np.float64)
        self.assert_dimensionally_invalid(cfunc, (ok, bad1D))
        self.assert_dimensionally_invalid(cfunc, (ok, bad2D))


class TestLinalgSolve(TestLinalgSystems):
    """
    Tests for np.linalg.solve.
    """

    @needs_lapack
    def test_linalg_solve(self):
        """
        Test np.linalg.solve
        """
        cfunc = jit(nopython=True)(solve_system)

        def check(a, b, **kwargs):
            expected = solve_system(a, b, **kwargs)
            got = cfunc(a, b, **kwargs)

            # check that the computed results are contig and in the same way
            self.assert_contig_sanity(got, "F")

            use_reconstruction = False
            # try plain match of the result first
            try:
                np.testing.assert_array_almost_equal_nulp(
                    got, expected, nulp=10)
            except AssertionError:
                # plain match failed, test by reconstruction
                use_reconstruction = True

            # If plain match fails then reconstruction is used,
            # this checks that AX ~= B.
            # Plain match can fail due to numerical fuzziness associated
            # with system size and conditioning, or more simply from
            # numpy using double precision routines for computation that
            # could be done in single precision (which is what numba does).
            # Therefore minor differences in results can appear due to
            # e.g. numerical roundoff being different between two precisions.
            if use_reconstruction:
                # check they are dimensionally correct
                self.assertEqual(got.shape, expected.shape)

                # check AX=B
                rec = np.dot(a, got)
                resolution = np.finfo(a.dtype).resolution
                np.testing.assert_allclose(
                    b,
                    rec,
                    rtol=10 * resolution,
                    atol=100 * resolution  # zeros tend to be fuzzy
                )

            # Ensure proper resource management
            with self.assertNoNRTLeak():
                cfunc(a, b, **kwargs)

        # test: prime size squares
        sizes = [(1, 1), (3, 3), (7, 7)]

        # There are a lot of combinations to test across all the different
        # dtypes, especially when type promotion comes into play, to
        # reduce the effort the dtype of "b" is cycled.
        cycle_dt = cycle(self.dtypes)

        # test loop
        for size, dtype, order in \
                product(sizes, self.dtypes, 'FC'):
            A = self.specific_sample_matrix(size, dtype, order)

            b_sizes = (1, 13)

            for b_size, b_order in product(b_sizes, 'FC'):
                # dtype for b
                dt = next(cycle_dt)

                # check 2D B
                B = self.specific_sample_matrix(
                    (A.shape[0], b_size), dt, b_order)
                check(A, B)

                # check 1D B
                tmp = B[:, 0].copy(order=b_order)
                check(A, tmp)

        # Test input validation
        ok = np.array([[1., 0.], [0., 1.]], dtype=np.float64)

        # check ok input is ok
        cfunc(ok, ok)

        # check bad inputs
        rn = "solve"

        # Wrong dtype
        bad = np.array([[1, 0], [0, 1]], dtype=np.int32)
        self.assert_wrong_dtype(rn, cfunc, (ok, bad))
        self.assert_wrong_dtype(rn, cfunc, (bad, ok))

        # Dimension issue
        bad = np.array([1, 0], dtype=np.float64)
        self.assert_wrong_dimensions(rn, cfunc, (bad, ok))

        # no nans or infs
        bad = np.array([[1., 0., ], [np.inf, np.nan]], dtype=np.float64)
        self.assert_no_nan_or_inf(cfunc, (ok, bad))
        self.assert_no_nan_or_inf(cfunc, (bad, ok))

        # check 1D is accepted for B (2D is done previously)
        # and then that anything of higher dimension raises
        ok_oneD = np.array([1., 2.], dtype=np.float64)
        cfunc(ok, ok_oneD)
        bad = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
        self.assert_wrong_dimensions_1D(rn, cfunc, (ok, bad))

        # check an invalid system raises (1D and 2D cases checked)
        bad1D = np.array([1.], dtype=np.float64)
        bad2D = np.array([[1.], [2.], [3.]], dtype=np.float64)
        self.assert_dimensionally_invalid(cfunc, (ok, bad1D))
        self.assert_dimensionally_invalid(cfunc, (ok, bad2D))

        # check that a singular system raises
        bad2D = self.specific_sample_matrix((2, 2), np.float64, 'C', rank=1)
        self.assert_raise_on_singular(cfunc, (bad2D, ok))


class TestLinalgPinv(TestLinalgBase):
    """
    Tests for np.linalg.pinv.
    """

    @needs_lapack
    def test_linalg_pinv(self):
        """
        Test np.linalg.pinv
        """
        cfunc = jit(nopython=True)(pinv_matrix)

        def check(a, **kwargs):
            expected = pinv_matrix(a, **kwargs)
            got = cfunc(a, **kwargs)

            # check that the computed results are contig and in the same way
            self.assert_contig_sanity(got, "F")

            use_reconstruction = False
            # try plain match of each array to np first

            try:
                np.testing.assert_array_almost_equal_nulp(
                    got, expected, nulp=10)
            except AssertionError:
                # plain match failed, test by reconstruction
                use_reconstruction = True

            # If plain match fails then reconstruction is used.
            # This can occur due to numpy using double precision
            # LAPACK when single can be used, this creates round off
            # problems. Also, if the matrix has machine precision level
            # zeros in its singular values then the singular vectors are
            # likely to vary depending on round off.
            if use_reconstruction:

                # check they are dimensionally correct
                self.assertEqual(got.shape, expected.shape)

                # check pinv(A)*A~=eye
                # if the problem is numerical fuzz then this will probably
                # work, if the problem is rank deficiency then it won't!
                rec = np.dot(got, a)
                try:
                    self.assert_is_identity_matrix(rec)
                except AssertionError:
                    # check A=pinv(pinv(A))
                    resolution = 5 * np.finfo(a.dtype).resolution
                    rec = cfunc(got)
                    np.testing.assert_allclose(
                        rec,
                        a,
                        rtol=10 * resolution,
                        atol=100 * resolution  # zeros tend to be fuzzy
                    )
                    if a.shape[0] >= a.shape[1]:
                        # if it is overdetermined or fully determined
                        # use numba lstsq function (which is type specific) to
                        # compute the inverse and check against that.
                        lstsq = jit(nopython=True)(lstsq_system)
                        lstsq_pinv = lstsq(
                            a, np.eye(
                                a.shape[0]).astype(
                                a.dtype), **kwargs)[0]
                        np.testing.assert_allclose(
                            got,
                            lstsq_pinv,
                            rtol=10 * resolution,
                            atol=100 * resolution  # zeros tend to be fuzzy
                        )
                    # check the 2 norm of the difference is small
                    self.assertLess(np.linalg.norm(got - expected), resolution)

            # Ensure proper resource management
            with self.assertNoNRTLeak():
                cfunc(a, **kwargs)

        # test: column vector, tall, wide, square, row vector
        # prime sizes
        sizes = [(7, 1), (11, 5), (5, 11), (3, 3), (1, 7)]

        # When required, a specified condition number
        specific_cond = 10.

        # test loop
        for size, dtype, order in \
                product(sizes, self.dtypes, 'FC'):
            # check a full rank matrix
            a = self.specific_sample_matrix(size, dtype, order)
            check(a)

            m, n = size
            if m != 1 and n != 1:
                # check a rank deficient matrix
                minmn = min(m, n)
                a = self.specific_sample_matrix(size, dtype, order,
                                                condition=specific_cond)
                rcond = 1. / specific_cond
                approx_half_rank_rcond = minmn * rcond
                check(a, rcond=approx_half_rank_rcond)

        rn = "pinv"

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
