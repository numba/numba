import gc
from itertools import product

import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.polynomial import polyutils as pu

from numba import jit, njit
from numba.tests.support import (TestCase, tag, needs_lapack,
                                 EnableNRTStatsMixin, MemoryLeakMixin)
from numba.core.errors import TypingError
import unittest


def roots_fn(p):
    return np.roots(p)

def polyadd(c1,c2):
    return poly.polyadd(c1,c2)

def polysub(c1,c2):
    return poly.polysub(c1,c2)

def polymul(c1,c2):
    return poly.polymul(c1,c2)

def trimseq(seq):
    return pu.trimseq(seq)


class TestPolynomialBase(EnableNRTStatsMixin, TestCase):
    """
    Provides setUp and common data/error modes for testing polynomial functions.
    """

    # supported dtypes
    dtypes = (np.float64, np.float32, np.complex128, np.complex64)

    def setUp(self):
        # Collect leftovers from previous test cases before checking for leaks
        gc.collect()
        super(TestPolynomialBase, self).setUp()

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

class TestPolynomial(MemoryLeakMixin, TestCase):

    def test_trimseq_basic(self):
        pyfunc = trimseq
        cfunc = njit(trimseq)
        for i in range(5):
            coefs = np.array([1] + [0]*i)
            self.assertPreciseEqual(pyfunc(coefs), cfunc(coefs))

    def test_trimseq_exception(self):
        cfunc = njit(trimseq)

        self.disable_leak_check()

        with self.assertRaises(TypingError) as raises:
            cfunc("abc")
        self.assertIn('The argument "seq" must be array-like',
                      str(raises.exception))

    def test_polyadd_basic(self):
        pyfunc = polyadd
        cfunc = njit(polyadd)
        def inputs():
            #basic
            for i in range(5):
                for j in range(5):
                    p1 = np.array([0]*i + [1])
                    p2 = np.array([0]*j + [1])
                    yield p1, p2
            #test lists, tuples, scalars
            yield [1, 2, 3], [1, 2, 3]
            yield [1, 2, 3], (1, 2, 3)
            yield (1, 2, 3), [1, 2, 3]
            yield [1, 2, 3], 3
            yield 3, (1, 2, 3)
            #test different dtypes
            yield np.array([1, 2, 3]), np.array([1.0, 2.0, 3.0])
            yield np.array([1j, 2j, 3j]), np.array([1.0, 2.0, 3.0])
            yield np.array([1, 2, 3]), np.array([1j, 2j, 3j])

        for p1, p2 in inputs():
            self.assertPreciseEqual(pyfunc(p1,p2), cfunc(p1,p2))

    def test_polyadd_exception(self):
        cfunc = njit(polyadd)

        self.disable_leak_check()

        with self.assertRaises(TypingError) as raises:
            cfunc("abc",np.array([1,2,3]))
        self.assertIn('The argument "c1" must be array-like',
                      str(raises.exception))
        
        with self.assertRaises(TypingError) as raises:
            cfunc(np.array([1,2,3]),"abc")
        self.assertIn('The argument "c2" must be array-like',
                      str(raises.exception))

    def test_polysub_basic(self):
        pyfunc = polysub
        cfunc = njit(polysub)
        def inputs():
            #basic
            for i in range(5):
                for j in range(5):
                    p1 = np.array([0]*i + [1])
                    p2 = np.array([0]*j + [1])
                    yield p1, p2
            #test lists, tuples, scalars
            yield [1, 2, 3], [1, 2, 3]
            yield [1, 2, 3], (1, 2, 3)
            yield (1, 2, 3), [1, 2, 3]
            yield [1, 2, 3], 3
            yield 3, (1, 2, 3)
            #test different dtypes
            yield np.array([1, 2, 3]), np.array([1.0, 2.0, 3.0])
            yield np.array([1j, 2j, 3j]), np.array([1.0, 2.0, 3.0])
            yield np.array([1, 2, 3]), np.array([1j, 2j, 3j])

        for p1, p2 in inputs():
            self.assertPreciseEqual(pyfunc(p1,p2), cfunc(p1,p2),
                                    ignore_sign_on_zero=True)

    def test_polysub_exception(self):
        cfunc = njit(polysub)

        self.disable_leak_check()

        with self.assertRaises(TypingError) as raises:
            cfunc("abc",np.array([1,2,3]))
        self.assertIn('The argument "c1" must be array-like',
                      str(raises.exception))
        
        with self.assertRaises(TypingError) as raises:
            cfunc(np.array([1,2,3]),"abc")
        self.assertIn('The argument "c2" must be array-like',
                      str(raises.exception))

    def test_polymul_basic(self):
        pyfunc = polymul
        cfunc = njit(polymul)
        def inputs():
            #basic
            for i in range(5):
                for j in range(5):
                    p1 = np.array([0]*i + [1])
                    p2 = np.array([0]*j + [1])
                    yield p1, p2
            #test lists, tuples, scalars
            yield [1, 2, 3], [1, 2, 3]
            yield [1, 2, 3], (1, 2, 3)
            yield (1, 2, 3), [1, 2, 3]
            yield [1, 2, 3], 3
            yield 3, (1, 2, 3)
            #test different dtypes
            yield np.array([1, 2, 3]), np.array([1.0, 2.0, 3.0])
            yield np.array([1j, 2j, 3j]), np.array([1.0, 2.0, 3.0])
            yield np.array([1, 2, 3]), np.array([1j, 2j, 3j])

        for p1, p2 in inputs():
            self.assertPreciseEqual(pyfunc(p1,p2), cfunc(p1,p2))

    def test_polymul_exception(self):
        cfunc = njit(polymul)

        self.disable_leak_check()

        with self.assertRaises(TypingError) as raises:
            cfunc("abc",np.array([1,2,3]))
        self.assertIn('The argument "c1" must be array-like',
                      str(raises.exception))
        
        with self.assertRaises(TypingError) as raises:
            cfunc(np.array([1,2,3]),"abc")
        self.assertIn('The argument "c2" must be array-like',
                      str(raises.exception))
