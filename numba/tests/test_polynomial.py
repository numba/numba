from __future__ import division, print_function

import gc
from itertools import product

import numpy as np

from numba import unittest_support as unittest
from numba import jit
from .support import TestCase, tag
from .test_linalg import needs_lapack


def roots_fn(p):
    return np.roots(p)


class TestPolynomialBase(TestCase):
    """
    Provides setUp and common data/error modes for testing polynomial functions.
    """

    # supported dtypes
    dtypes = (np.float64, np.float32, np.complex128, np.complex64)

    def setUp(self):
        # Collect leftovers from previous test cases before checking for leaks
        gc.collect()

    def assert_error(self, cfunc, args, msg, err=ValueError):
        with self.assertRaises(err) as raises:
            cfunc(*args)
        self.assertIn(msg, str(raises.exception))

    def assert_1d_input(self, cfunc, args):
        msg = "Input must be a 1d array."
        self.assert_error(cfunc, args, msg)


class TestPoly1D(TestPolynomialBase):

    def assert_no_domain_change(self, name, cfunc, args):
        msg = name + "() argument must not cause a domain change."
        self.assert_error(cfunc, args, msg)

    @needs_lapack
    def test_roots(self):

        cfunc = jit(nopython=True)(roots_fn)

        default_resolution = np.finfo(np.float64).resolution

        def check(a, **kwargs):
            expected = roots_fn(a, **kwargs)
            got = cfunc(a, **kwargs)

            # eigen decomposition used so type specific impl
            # will be used in numba whereas a wide type impl
            # will be used in numpy, so compare using a more
            # fuzzy comparator

            if a.dtype in self.dtypes:
                resolution = np.finfo(a.dtype).resolution
            else:
                # this is for integer types when roots() will cast to float64
                resolution = default_resolution

            np.testing.assert_allclose(
                expected,
                got,
                rtol=10 * resolution,
                atol=100 * resolution  # zeros tend to be fuzzy
            )

            # Ensure proper resource management
            with self.assertNoNRTLeak():
                cfunc(a, **kwargs)

        # test vectors in real space
        # contrived examples to trip branches
        r_vectors = (
            np.array([1]),
            np.array([1, 3, 2]),
            np.array([0, 0, 0]),
            np.array([1, 6, 11, 6]),
            np.array([0, 0, 0, 1, 3, 2]),
            np.array([1, 1, 0, 0, 0]),
            np.array([0, 0, 1, 0, 0, 0])
        )

        # test loop real space
        for v, dtype in \
                product(r_vectors, [np.int32, np.int64] + list(self.dtypes)):
            a = v.astype(dtype)
            check(a)

        c_vectors = (
            np.array([1 + 1j]),
            np.array([1, 3 + 1j, 2]),
            np.array([0, 0 + 0j, 0]),
            np.array([1, 6 + 1j, 11, 6]),
            np.array([0, 0, 0, 1 + 1j, 3, 2]),
            np.array([1 + 1j, 1, 0, 0, 0]),
            np.array([0, 0, 1 + 1j, 0, 0, 0])
        )

        # test loop complex space
        for v, dtype in product(c_vectors, self.dtypes[2:]):
            a = v.astype(dtype)
            check(a)

        # check input with dimension > 1 raises
        self.assert_1d_input(cfunc, (np.arange(4.).reshape(2, 2),))

        # check real input with complex roots raises
        x = np.array([7., 2., 0., 1.])
        self.assert_no_domain_change("eigvals", cfunc, (x,))
        # but works fine if type conv to complex first
        cfunc(x.astype(np.complex128))
