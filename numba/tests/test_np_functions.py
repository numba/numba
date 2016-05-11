# Tests numpy methods of <class 'function'>
from __future__ import print_function, absolute_import, division

import itertools
import math
import sys

import numpy as np

from numba import unittest_support as unittest
from numba.compiler import compile_isolated, Flags, utils
from numba import jit, typeof, types
from numba.numpy_support import version as np_version
from .support import TestCase, CompilationCache

no_pyobj_flags = Flags()
no_pyobj_flags.set("nrt")


def sinc(x):
    return np.sinc(x)

def angle1(x):
    return np.angle(x)

def angle2(x, deg):
    return np.angle(x, deg)

def diff1(a):
    return np.diff(a)

def diff2(a, n):
    return np.diff(a, n)

def bincount1(a):
    return np.bincount(a)

def bincount2(a, w):
    return np.bincount(a, weights=w)

def searchsorted(a, v):
    return np.searchsorted(a, v)

def digitize(*args):
    return np.digitize(*args)

def histogram(*args):
    return np.histogram(*args)


class TestNPFunctions(TestCase):
    """
    Tests for various Numpy functions.
    """

    def setUp(self):
        self.ccache = CompilationCache()
        self.rnd = np.random.RandomState(42)

    def run_unary(self, pyfunc, x_types, x_values, flags=no_pyobj_flags,
                  func_extra_types=None, func_extra_args=None,
                  ignore_sign_on_zero=False, abs_tol=None, **kwargs):
        """
        Runs tests for a unary function operating in the numerical real space.

        Parameters
        ----------
        pyfunc : a python function definition holding that calls the numpy
                 functions to be tested.
        x_types: the types of the values being tested, see numba.types
        x_values: the numerical values of the values to be tested
        flags: flags to pass to the CompilationCache::ccache::compile function
        func_extra_types: the types of additional arguments to the numpy
                          function
        func_extra_args:  additional arguments to the numpy function
        ignore_sign_on_zero: boolean as to whether to allow zero values
        with incorrect signs to be considered equal
        prec: the required precision match, see assertPreciseEqual

        Notes:
        ------
        x_types and x_values must have the same length

        """
        for tx, vx in zip(x_types, x_values):
            if func_extra_args is None:
                func_extra_types = func_extra_args = [()]
            for xtypes, xargs in zip(func_extra_types, func_extra_args):
                cr = self.ccache.compile(pyfunc, (tx,) + xtypes,
                                         flags=flags)
                cfunc = cr.entry_point
                got = cfunc(vx, *xargs)
                expected = pyfunc(vx, *xargs)
                try:
                    scalty = tx.dtype
                except AttributeError:
                    scalty = tx
                prec = ('single'
                        if scalty in (types.float32, types.complex64)
                        else 'double')
                msg = 'for input %r with prec %r' % (vx, prec)
                self.assertPreciseEqual(got, expected,
                                        prec=prec,
                                        msg=msg,
                                        ignore_sign_on_zero=
                                        ignore_sign_on_zero,
                                        abs_tol=abs_tol, **kwargs)

    def test_sinc(self):
        """
        Tests the sinc() function.
        This test is purely to assert numerical computations are correct.
        """

        # Ignore sign of zeros, this will need masking depending on numpy
        # version once the fix to numpy complex division is in upstream
        # See: https://github.com/numpy/numpy/pull/6699
        isoz = True

        # Testing sinc(1.) leads to sin(pi)/pi, which is below machine
        # precision in practice on most machines. Small floating point
        # differences in sin() etc. may lead to large differences in the result
        # that are at a range that is inaccessible using standard width
        # floating point representations.
        # e.g. Assume float64 type.
        # sin(pi) ~= 1e-16, but should be zero
        # sin(pi)/pi ~= 1e-17, should be zero, error carried from above
        # float64 has log10(2^53)~=15.9 digits of precision and the magnitude
        # change in the alg is > 16  digits (1.0...0 -> 0.0...0),
        # so comparison via ULP is invalid.
        # We therefore opt to assume that values under machine precision are
        # equal in this case.
        tol = "eps"

        pyfunc = sinc

        def check(x_types, x_values, **kwargs):
            self.run_unary(pyfunc, x_types, x_values,
                                ignore_sign_on_zero=isoz, abs_tol=tol,
                                **kwargs)

        # real domain scalar context
        x_values = [1., -1., 0.0, -0.0, 0.5, -0.5, 5, -5, 5e-21, -5e-21]
        x_types = [types.float32, types.float64] * (len(x_values) // 2)
        check(x_types, x_values)

        # real domain vector context
        x_values = [np.array(x_values, dtype=np.float64)]
        x_types = [typeof(v) for v in x_values]
        check(x_types, x_values)

        # complex domain scalar context
        x_values = [1.+0j, -1+0j, 0.0+0.0j, -0.0+0.0j, 0+1j, 0-1j, 0.5+0.0j,
                    -0.5+0.0j, 0.5+0.5j, -0.5-0.5j, 5+5j, -5-5j,
                    # the following are to test sin(x)/x for small x
                    5e-21+0j, -5e-21+0j, 5e-21j, +(0-5e-21j)
                    ]
        x_types = [types.complex64, types.complex128] * (len(x_values) // 2)
        check(x_types, x_values, ulps=2)

        # complex domain vector context
        x_values = [np.array(x_values, dtype=np.complex128)]
        x_types = [typeof(v) for v in x_values]
        check(x_types, x_values, ulps=2)


    def test_angle(self, flags=no_pyobj_flags):
        """
        Tests the angle() function.
        This test is purely to assert numerical computations are correct.
        """
        pyfunc1 = angle1
        pyfunc2 = angle2

        def check(x_types, x_values):
            # angle(x)
            self.run_unary(pyfunc1, x_types, x_values)
            # angle(x, deg)
            xtra_values = [(True,), (False,)]
            xtra_types = [(types.bool_,)] * len(xtra_values)
            self.run_unary(pyfunc2, x_types, x_values,
                           func_extra_types=xtra_types,
                           func_extra_args=xtra_values,)

        # real domain scalar context
        x_values = [1., -1., 0.0, -0.0, 0.5, -0.5, 5, -5]
        x_types = [types.float32, types.float64] * (len(x_values) // 2 + 1)
        check(x_types, x_values)

        # real domain vector context
        x_values = [np.array(x_values, dtype=np.float64)]
        x_types = [typeof(v) for v in x_values]
        check(x_types, x_values)

        # complex domain scalar context
        x_values = [1.+0j, -1+0j, 0.0+0.0j, -0.0+0.0j, 1j, -1j, 0.5+0.0j,
                    -0.5+0.0j, 0.5+0.5j, -0.5-0.5j, 5+5j, -5-5j]
        x_types = [types.complex64, types.complex128] * (len(x_values) // 2 + 1)
        check(x_types, x_values)

        # complex domain vector context
        x_values = np.array(x_values)
        x_types = [types.complex64, types.complex128]
        check(x_types, x_values)


    def diff_arrays(self):
        """
        Some test arrays for np.diff()
        """
        a = np.arange(12) ** 3
        yield a
        b = a.reshape((3, 4))
        yield b
        c = np.arange(24).reshape((3, 2, 4)) ** 3
        yield c

    def test_diff1(self):
        pyfunc = diff1
        cfunc = jit(nopython=True)(pyfunc)
        for arr in self.diff_arrays():
            expected = pyfunc(arr)
            got = cfunc(arr)
            self.assertPreciseEqual(expected, got)

        # 0-dim array
        a = np.array(42)
        with self.assertTypingError():
            cfunc(a)

    def test_diff2(self):
        pyfunc = diff2
        cfunc = jit(nopython=True)(pyfunc)
        for arr in self.diff_arrays():
            size = arr.shape[-1]
            for n in (0, 1, 2, 3, size - 1, size, size + 1, 421):
                expected = pyfunc(arr, n)
                got = cfunc(arr, n)
                self.assertPreciseEqual(expected, got)

        # 0-dim array
        arr = np.array(42)
        with self.assertTypingError():
            cfunc(arr, 1)
        # Invalid `n`
        arr = np.arange(10)
        for n in (-1, -2, -42):
            with self.assertRaises(ValueError) as raises:
                cfunc(arr, n)
            self.assertIn("order must be non-negative", str(raises.exception))

    def bincount_sequences(self):
        """
        Some test sequences for np.bincount()
        """
        a = [1, 2, 5, 2, 3, 20]
        b = np.array([5, 8, 42, 5])
        c = self.rnd.randint(0, 100, size=300).astype(np.int8)
        return (a, b, c)

    def test_bincount1(self):
        pyfunc = bincount1
        cfunc = jit(nopython=True)(pyfunc)
        for seq in self.bincount_sequences():
            expected = pyfunc(seq)
            got = cfunc(seq)
            self.assertPreciseEqual(expected, got)

        # Negative input
        with self.assertRaises(ValueError) as raises:
            cfunc([2, -1])
        self.assertIn("first argument must be non-negative",
                      str(raises.exception))

    def test_bincount2(self):
        pyfunc = bincount2
        cfunc = jit(nopython=True)(pyfunc)
        for seq in self.bincount_sequences():
            w = [math.sqrt(x) - 2 for x in seq]
            # weights as list, then array
            for weights in (w, np.array(w)):
                expected = pyfunc(seq, weights)
                got = cfunc(seq, weights)
                self.assertPreciseEqual(expected, got)

        # Negative input
        with self.assertRaises(ValueError) as raises:
            cfunc([2, -1], [0, 0])
        self.assertIn("first argument must be non-negative",
                      str(raises.exception))

        # Mismatching input sizes
        with self.assertRaises(ValueError) as raises:
            cfunc([2, -1], [0])
        self.assertIn("weights and list don't have the same length",
                      str(raises.exception))

    def test_searchsorted(self):
        pyfunc = searchsorted
        cfunc = jit(nopython=True)(pyfunc)

        def check(a, v):
            expected = pyfunc(a, v)
            got = cfunc(a, v)
            self.assertPreciseEqual(expected, got)

        # First with integer values (no NaNs)
        bins = np.arange(5) ** 2
        values = np.arange(20) - 1

        for a in (bins, list(bins)):
            # Scalar values
            for v in values:
                check(a, v)
            # Array values
            for v in (values, values.reshape((4, 5))):
                check(a, v)
            # Sequence values
            check(a, list(values))

        # Second with float values (including NaNs)
        bins = np.float64(list(bins) + [float('nan')] * 7) / 2.0
        values = np.arange(20) - 0.5

        for a in (bins, list(bins)):
            # Scalar values
            for v in values:
                check(a, v)
            # Array values
            for v in (values, values.reshape((4, 5))):
                check(a, v)
            # Sequence values
            check(a, list(values))

    def test_digitize(self):
        pyfunc = digitize
        cfunc = jit(nopython=True)(pyfunc)

        def check(*args):
            expected = pyfunc(*args)
            got = cfunc(*args)
            self.assertPreciseEqual(expected, got)

        values = np.float64((0, 0.99, 1, 4.4, 4.5, 7, 8, 9, 9.5,
                             float('inf'), float('-inf'), float('nan')))
        assert len(values) == 12
        self.rnd.shuffle(values)

        bins1 = np.float64([1, 3, 4.5, 8])
        bins2 = np.float64([1, 3, 4.5, 8, float('inf'), float('-inf')])
        bins3 = np.float64([1, 3, 4.5, 8, float('inf'), float('-inf')]
                           + [float('nan')] * 10)
        if np_version >= (1, 10):
            all_bins = [bins1, bins2, bins3]
            xs = [values, values.reshape((3, 4))]
        else:
            # Numpy < 1.10 had trouble with NaNs and N-d arrays
            all_bins = [bins1, bins2]
            xs = [values]


        # 2-ary digitize()
        for bins in all_bins:
            bins.sort()
            for x in xs:
                check(x, bins)
                check(x, bins[::-1])

        # 3-ary digitize()
        for bins in all_bins:
            bins.sort()
            for right in (True, False):
                check(values, bins, right)
                check(values, bins[::-1], right)

        # Sequence input
        check(list(values), bins1)

    def test_histogram(self):
        pyfunc = histogram
        cfunc = jit(nopython=True)(pyfunc)

        def check(*args):
            pyhist, pybins = pyfunc(*args)
            chist, cbins = cfunc(*args)
            self.assertPreciseEqual(pyhist, chist)
            # There can be a slight discrepancy in the linspace() result
            # when `bins` is an integer...
            self.assertPreciseEqual(pybins, cbins, prec='double', ulps=2)

        def check_values(values):
            # Explicit bins array
            # (note Numpy seems to not support NaN bins)
            bins = np.float64([1, 3, 4.5, 8])
            check(values, bins)
            check(values.reshape((3, 4)), bins)

            # Explicit number of bins
            check(values, 7)

            # Explicit number of bins and bins range
            check(values, 7, (1.0, 13.5))

            # Implicit bins=10
            check(values)

        values = np.float64((0, 0.99, 1, 4.4, 4.5, 7, 8,
                             9, 9.5, 42.5, -1.0, -0.0))
        assert len(values) == 12
        self.rnd.shuffle(values)

        check_values(values)
