# Tests numpy methods of <class 'function'>
from __future__ import print_function, absolute_import, division

import itertools
import math
import sys
from functools import partial

import numpy as np

from numba import unittest_support as unittest
from numba.compiler import compile_isolated, Flags, utils
from numba import jit, typeof, types
from numba.numpy_support import version as np_version
from numba.errors import TypingError
from .support import TestCase, CompilationCache, MemoryLeakMixin
from .matmul_usecase import needs_blas

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

def searchsorted_left(a, v):
    return np.searchsorted(a, v, side='left')

def searchsorted_right(a, v):
    return np.searchsorted(a, v, side='right')

def digitize(*args):
    return np.digitize(*args)

def histogram(*args):
    return np.histogram(*args)

def machar(*args):
    return np.MachAr()

def iinfo(*args):
    return np.iinfo(*args)

def finfo(*args):
    return np.finfo(*args)

def finfo_machar(*args):
    return np.finfo(*args).machar

def correlate(a, v):
    return np.correlate(a, v)

def convolve(a, v):
    return np.convolve(a, v)

def tri_n(N):
    return np.tri(N)

def tri_n_m(N, M=None):
    return np.tri(N, M)

def tri_n_k(N, k=0):
    return np.tri(N, k)

def tri_n_m_k(N, M=None, k=0):
    return np.tri(N, M, k)

def tril_m(m):
    return np.tril(m)

def tril_m_k(m, k=0):
    return np.tril(m, k)

def triu_m(m):
    return np.triu(m)

def triu_m_k(m, k=0):
    return np.triu(m, k)

def vander(x, N=None, increasing=False):
    return np.vander(x, N, increasing)

def partition(a, kth):
    return np.partition(a, kth)

def cov(m, y=None, rowvar=True, bias=False, ddof=None):
    return np.cov(m, y, rowvar, bias, ddof)

def ediff1d(ary, to_end=None, to_begin=None):
    return np.ediff1d(ary, to_end, to_begin)


class TestNPFunctions(MemoryLeakMixin, TestCase):
    """
    Tests for various Numpy functions.
    """

    def setUp(self):
        super(TestNPFunctions, self).setUp()
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

    def test_diff2_exceptions(self):
        pyfunc = diff2
        cfunc = jit(nopython=True)(pyfunc)

        # Exceptions leak references
        self.disable_leak_check()

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

    def test_bincount1_exceptions(self):
        pyfunc = bincount1
        cfunc = jit(nopython=True)(pyfunc)

        # Exceptions leak references
        self.disable_leak_check()

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

    def test_bincount2_exceptions(self):
        pyfunc = bincount2
        cfunc = jit(nopython=True)(pyfunc)

        # Exceptions leak references
        self.disable_leak_check()

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

        pyfunc_left = searchsorted_left
        cfunc_left = jit(nopython=True)(pyfunc_left)

        pyfunc_right = searchsorted_right
        cfunc_right = jit(nopython=True)(pyfunc_right)

        def check(a, v):
            expected = pyfunc(a, v)
            got = cfunc(a, v)
            self.assertPreciseEqual(expected, got)

            expected = pyfunc_left(a, v)
            got = cfunc_left(a, v)
            self.assertPreciseEqual(expected, got)

            expected = pyfunc_right(a, v)
            got = cfunc_right(a, v)
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

        # nonsense value for 'side' raises TypingError
        def bad_side(a, v):
            return np.searchsorted(a, v, side='nonsense')
        cfunc = jit(nopython=True)(bad_side)
        with self.assertTypingError():
            cfunc([1,2], 1)

        # non-constant value for 'side' raises TypingError
        def nonconst_side(a, v, side='left'):
            return np.searchsorted(a, v, side=side)
        cfunc = jit(nopython=True)(nonconst_side)
        with self.assertTypingError():
            cfunc([1,2], 1, side='right')

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

    def _test_correlate_convolve(self, pyfunc):
        cfunc = jit(nopython=True)(pyfunc)
        # only 1d arrays are accepted, test varying lengths
        # and varying dtype
        lengths = (1, 2, 3, 7)
        dts = [np.int8, np.int32, np.int64, np.float32, np.float64,
               np.complex64, np.complex128]

        for dt1, dt2, n, m in itertools.product(dts, dts, lengths, lengths):
            a = np.arange(n, dtype=dt1)
            v = np.arange(m, dtype=dt2)

            if np.issubdtype(dt1, np.complexfloating):
                a = (a + 1j * a).astype(dt1)
            if np.issubdtype(dt2, np.complexfloating):
                v = (v + 1j * v).astype(dt2)

            expected = pyfunc(a, v)
            got = cfunc(a, v)
            self.assertPreciseEqual(expected, got)

        _a = np.arange(12).reshape(4, 3)
        _b = np.arange(12)
        for x, y in [(_a, _b), (_b, _a)]:
            with self.assertRaises(TypingError) as raises:
                cfunc(x, y)
            msg = 'only supported on 1D arrays'
            self.assertIn(msg, str(raises.exception))

    def test_correlate(self):
        self._test_correlate_convolve(correlate)
        # correlate supports 0 dimension arrays
        _a = np.ones(shape=(0,))
        _b = np.arange(5)
        cfunc = jit(nopython=True)(correlate)
        for x, y in [(_a, _b), (_b, _a), (_a, _a)]:
            expected = correlate(x, y)
            got = cfunc(x, y)
            self.assertPreciseEqual(expected, got)

    def test_convolve(self):
        self._test_correlate_convolve(convolve)

    def test_convolve_exceptions(self):
        # Exceptions leak references
        self.disable_leak_check()

        # convolve raises if either array has a 0 dimension
        _a = np.ones(shape=(0,))
        _b = np.arange(5)
        cfunc = jit(nopython=True)(convolve)
        for x, y in [(_a, _b), (_b, _a)]:
            with self.assertRaises(ValueError) as raises:
                cfunc(x, y)
            if len(x) == 0:
                self.assertIn("'a' cannot be empty", str(raises.exception))
            else:
                self.assertIn("'v' cannot be empty", str(raises.exception))

    def _check_output(self, pyfunc, cfunc, params, abs_tol=None):
        expected = pyfunc(**params)
        got = cfunc(**params)
        self.assertPreciseEqual(expected, got, abs_tol=abs_tol)

    def test_vander_basic(self):
        pyfunc = vander
        cfunc = jit(nopython=True)(pyfunc)
        _check_output = partial(self._check_output, pyfunc, cfunc)

        def _check(x):
            n_choices = [None, 0, 1, 2, 3, 4]
            increasing_choices = [True, False]

            # N and increasing defaulted
            params = {'x': x}
            _check_output(params)

            # N provided and increasing defaulted
            for n in n_choices:
                params = {'x': x, 'N': n}
                _check_output(params)

            # increasing provided and N defaulted:
            for increasing in increasing_choices:
                params = {'x': x, 'increasing': increasing}
                _check_output(params)

            # both n and increasing supplied
            for n in n_choices:
                for increasing in increasing_choices:
                    params = {'x': x, 'N': n, 'increasing': increasing}
                    _check_output(params)

        _check(np.array([1, 2, 3, 5]))
        _check(np.arange(7) - 10.5)
        _check(np.linspace(3, 10, 5))
        _check(np.array([1.2, np.nan, np.inf, -np.inf]))
        _check(np.array([]))
        _check(np.arange(-5, 5) - 0.3)

        # # boolean array
        _check(np.array([True] * 5 + [False] * 4))

        # cycle through dtypes to check type promotion a la numpy
        for dtype in np.int32, np.int64, np.float32, np.float64:
            _check(np.arange(10, dtype=dtype))

        # non array inputs
        _check([0, 1, 2, 3])
        _check((4, 5, 6, 7))
        _check((0.0, 1.0, 2.0))
        _check(())

        # edge cases
        _check((3, 4.444, 3.142))
        _check((True, False, 4))

    def test_vander_exceptions(self):
        pyfunc = vander
        cfunc = jit(nopython=True)(pyfunc)

        # Exceptions leak references
        self.disable_leak_check()

        x = np.arange(5) - 0.5

        def _check_n(N):
            with self.assertTypingError() as raises:
                cfunc(x, N=N)
            assert "Second argument N must be None or an integer" in str(raises.exception)

        for N in 1.1, True, np.inf, [1, 2]:
            _check_n(N)

        with self.assertRaises(ValueError) as raises:
            cfunc(x, N=-1)
        assert "Negative dimensions are not allowed" in str(raises.exception)

        def _check_1d(x):
            with self.assertRaises(ValueError) as raises:
                cfunc(x)
            self.assertEqual("x must be a one-dimensional array or sequence.", str(raises.exception))

        x = np.arange(27).reshape((3, 3, 3))
        _check_1d(x)

        x = ((2, 3), (4, 5))
        _check_1d(x)

    def test_tri_n_basic(self):
        pyfunc = tri_n
        cfunc = jit(nopython=True)(pyfunc)
        _check = partial(self._check_output, pyfunc, cfunc)

        def n_variations():
            return np.arange(-4, 8)  # number of rows

        # N supplied, M and k defaulted
        for n in n_variations():
            params = {'N': n}
            _check(params)

    def test_tri_n_m_basic(self):
        pyfunc = tri_n_m
        cfunc = jit(nopython=True)(pyfunc)
        _check = partial(self._check_output, pyfunc, cfunc)

        def n_variations():
            return np.arange(-4, 8)  # number of rows

        def m_variations():
            return itertools.chain.from_iterable(([None], range(-5, 9)))  # number of columns

        # N supplied, M and k defaulted
        for n in n_variations():
            params = {'N': n}
            _check(params)

        # N and M supplied, k defaulted
        for n in n_variations():
            for m in m_variations():
                params = {'N': n, 'M': m}
                _check(params)

    def test_tri_n_k_basic(self):
        pyfunc = tri_n_k
        cfunc = jit(nopython=True)(pyfunc)
        _check = partial(self._check_output, pyfunc, cfunc)

        def n_variations():
            return np.arange(-4, 8)  # number of rows

        def k_variations():
            return np.arange(-10, 10)  # offset

        # N supplied, M and k defaulted
        for n in n_variations():
            params = {'N': n}
            _check(params)

        # N and k supplied, M defaulted
        for n in n_variations():
            for k in k_variations():
                params = {'N': n, 'k': k}
                _check(params)

    def test_tri_n_m_k_basic(self):
        pyfunc = tri_n_m_k
        cfunc = jit(nopython=True)(pyfunc)
        _check = partial(self._check_output, pyfunc, cfunc)

        def n_variations():
            return np.arange(-4, 8)  # number of rows

        def m_variations():
            return itertools.chain.from_iterable(([None], range(-5, 9)))  # number of columns

        def k_variations():
            return np.arange(-10, 10)  # offset

        # N supplied, M and k defaulted
        for n in n_variations():
            params = {'N': n}
            _check(params)

        # N and M supplied, k defaulted
        for n in n_variations():
            for m in m_variations():
                params = {'N': n, 'M': m}
                _check(params)

        # N and k supplied, M defaulted
        for n in n_variations():
            for k in k_variations():
                params = {'N': n, 'k': k}
                _check(params)

        # N, M and k supplied
        for n in n_variations():
            for k in k_variations():
                for m in m_variations():
                    params = {'N': n, 'M': m, 'k': k}
                    _check(params)

    def test_tri_exceptions(self):
        pyfunc = tri_n_m_k
        cfunc = jit(nopython=True)(pyfunc)

        # Exceptions leak references
        self.disable_leak_check()

        def _check(k):
            with self.assertTypingError() as raises:
                cfunc(5, 6, k=k)
            assert "k must be an integer" in str(raises.exception)

        for k in 1.5, True, np.inf, [1, 2]:
            _check(k)

    def _triangular_matrix_tests_m(self, pyfunc):
        cfunc = jit(nopython=True)(pyfunc)

        def _check(arr):
            expected = pyfunc(arr)
            got = cfunc(arr)
            # TODO: Contiguity of result not consistent with numpy
            self.assertEqual(got.dtype, expected.dtype)
            np.testing.assert_array_equal(got, expected)

        return self._triangular_matrix_tests_inner(self, pyfunc, _check)

    def _triangular_matrix_tests_m_k(self, pyfunc):
        cfunc = jit(nopython=True)(pyfunc)

        def _check(arr):
            for k in itertools.chain.from_iterable(([None], range(-10, 10))):
                if k is None:
                    params = {}
                else:
                    params = {'k': k}
                expected = pyfunc(arr, **params)
                got = cfunc(arr, **params)
                # TODO: Contiguity of result not consistent with numpy
                self.assertEqual(got.dtype, expected.dtype)
                np.testing.assert_array_equal(got, expected)

        return self._triangular_matrix_tests_inner(self, pyfunc, _check)

    @staticmethod
    def _triangular_matrix_tests_inner(self, pyfunc, _check):

        def check_odd(a):
            _check(a)
            a = a.reshape((9, 7))
            _check(a)
            a = a.reshape((7, 1, 3, 3))
            _check(a)
            _check(a.T)

        def check_even(a):
            _check(a)
            a = a.reshape((4, 16))
            _check(a)
            a = a.reshape((4, 2, 2, 4))
            _check(a)
            _check(a.T)

        check_odd(np.arange(63) + 10.5)
        check_even(np.arange(64) - 10.5)

        # edge cases
        _check(np.arange(360).reshape(3, 4, 5, 6))
        _check(np.array([]))
        _check(np.arange(9).reshape((3, 3))[::-1])
        _check(np.arange(9).reshape((3, 3), order='F'))

        arr = (np.arange(64) - 10.5).reshape((4, 2, 2, 4))
        _check(arr)
        _check(np.asfortranarray(arr))

    def _triangular_matrix_exceptions(self, pyfunc):
        cfunc = jit(nopython=True)(pyfunc)

        # Exceptions leak references
        self.disable_leak_check()

        a = np.ones((5, 6))
        with self.assertTypingError() as raises:
            cfunc(a, k=1.5)
        assert "k must be an integer" in str(raises.exception)

    def test_tril_basic(self):
        self._triangular_matrix_tests_m(tril_m)
        self._triangular_matrix_tests_m_k(tril_m_k)

    def test_tril_exceptions(self):
        self._triangular_matrix_exceptions(tril_m_k)

    def test_triu_basic(self):
        self._triangular_matrix_tests_m(triu_m)
        self._triangular_matrix_tests_m_k(triu_m_k)

    def test_triu_exceptions(self):
        self._triangular_matrix_exceptions(triu_m_k)

    def partition_sanity_check(self, pyfunc, cfunc, a, kth):
        # as NumPy uses a different algorithm, we do not expect to match outputs exactly...
        expected = pyfunc(a, kth)
        got = cfunc(a, kth)

        # ... but we do expect the unordered collection of elements up to kth to tie out
        self.assertPreciseEqual(np.unique(expected[:kth]), np.unique(got[:kth]))

        # ... likewise the unordered collection of elements from kth onwards
        self.assertPreciseEqual(np.unique(expected[kth:]), np.unique(got[kth:]))

    def test_partition_fuzz(self):
        # inspired by the test of the same name in:
        # https://github.com/numpy/numpy/blob/043a840/numpy/core/tests/test_multiarray.py
        pyfunc = partition
        cfunc = jit(nopython=True)(pyfunc)

        for j in range(10, 30):
            for i in range(1, j - 2):
                d = np.arange(j)
                self.rnd.shuffle(d)
                d = d % self.rnd.randint(2, 30)
                idx = self.rnd.randint(d.size)
                kth = [0, idx, i, i + 1, -idx, -i]  # include some negative kth's
                tgt = np.sort(d)[kth]
                self.assertPreciseEqual(cfunc(d, kth)[kth], tgt)  # a -> array
                self.assertPreciseEqual(cfunc(d.tolist(), kth)[kth], tgt)  # a -> list
                self.assertPreciseEqual(cfunc(tuple(d.tolist()), kth)[kth], tgt)  # a -> tuple

                for k in kth:
                    self.partition_sanity_check(pyfunc, cfunc, d, k)

    def test_partition_exception_out_of_range(self):
        # inspired by the test of the same name in:
        # https://github.com/numpy/numpy/blob/043a840/numpy/core/tests/test_multiarray.py
        pyfunc = partition
        cfunc = jit(nopython=True)(pyfunc)

        # Exceptions leak references
        self.disable_leak_check()

        # Test out of range values in kth raise an error
        a = np.arange(10)

        def _check(a, kth):
            with self.assertRaises(ValueError) as e:
                cfunc(a, kth)
            assert str(e.exception) == "kth out of bounds"

        _check(a, 10)
        _check(a, -11)
        _check(a, (3, 30))

    def test_partition_exception_non_integer_kth(self):
        # inspired by the test of the same name in:
        # https://github.com/numpy/numpy/blob/043a840/numpy/core/tests/test_multiarray.py
        pyfunc = partition
        cfunc = jit(nopython=True)(pyfunc)

        # Exceptions leak references
        self.disable_leak_check()

        def _check(a, kth):
            with self.assertTypingError() as raises:
                cfunc(a, kth)
            self.assertIn("Partition index must be integer", str(raises.exception))

        a = np.arange(10)
        _check(a, 9.0)
        _check(a, (3.3, 4.4))
        _check(a, np.array((1, 2, np.nan)))

    def test_partition_exception_a_not_array_like(self):
        pyfunc = partition
        cfunc = jit(nopython=True)(pyfunc)

        # Exceptions leak references
        self.disable_leak_check()

        def _check(a, kth):
            with self.assertTypingError() as raises:
                cfunc(a, kth)
            self.assertIn('The first argument must be an array-like', str(raises.exception))

        _check(4, 0)
        _check('Sausages', 0)

    def test_partition_exception_a_zero_dim(self):
        pyfunc = partition
        cfunc = jit(nopython=True)(pyfunc)

        # Exceptions leak references
        self.disable_leak_check()

        def _check(a, kth):
            with self.assertTypingError() as raises:
                cfunc(a, kth)
            self.assertIn('The first argument must be at least 1-D (found 0-D)', str(raises.exception))

        _check(np.array(1), 0)

    def test_partition_exception_kth_multi_dimensional(self):
        pyfunc = partition
        cfunc = jit(nopython=True)(pyfunc)

        # Exceptions leak references
        self.disable_leak_check()

        def _check(a, kth):
            with self.assertRaises(ValueError) as raises:
                cfunc(a, kth)
            self.assertIn('kth must be scalar or 1-D', str(raises.exception))

        _check(np.arange(10), kth=np.arange(6).reshape(3, 2))

    def test_partition_empty_array(self):
        # inspired by the test of the same name in:
        # https://github.com/numpy/numpy/blob/043a840/numpy/core/tests/test_multiarray.py
        pyfunc = partition
        cfunc = jit(nopython=True)(pyfunc)

        def check(a, kth=0):
            expected = pyfunc(a, kth)
            got = cfunc(a, kth)
            self.assertPreciseEqual(expected, got)

        # check axis handling for multidimensional empty arrays
        a = np.array([])
        a.shape = (3, 2, 1, 0)

        # include this with some other empty data structures
        for arr in a, (), np.array([]):
            check(arr)

    def test_partition_basic(self):
        # inspired by the test of the same name in:
        # https://github.com/numpy/numpy/blob/043a840/numpy/core/tests/test_multiarray.py
        pyfunc = partition
        cfunc = jit(nopython=True)(pyfunc)

        d = np.array([])
        got = cfunc(d, 0)
        self.assertPreciseEqual(d, got)

        d = np.ones(1)
        got = cfunc(d, 0)
        self.assertPreciseEqual(d, got)

        # kth not modified
        kth = np.array([30, 15, 5])
        okth = kth.copy()
        cfunc(np.arange(40), kth)
        self.assertPreciseEqual(kth, okth)

        for r in ([2, 1], [1, 2], [1, 1]):
            d = np.array(r)
            tgt = np.sort(d)
            for k in 0, 1:
                self.assertPreciseEqual(cfunc(d, k)[k], tgt[k])
                self.partition_sanity_check(pyfunc, cfunc, d, k)

        for r in ([3, 2, 1], [1, 2, 3], [2, 1, 3], [2, 3, 1],
                  [1, 1, 1], [1, 2, 2], [2, 2, 1], [1, 2, 1]):
            d = np.array(r)
            tgt = np.sort(d)
            for k in 0, 1, 2:
                self.assertPreciseEqual(cfunc(d, k)[k], tgt[k])
                self.partition_sanity_check(pyfunc, cfunc, d, k)

        d = np.ones(50)
        self.assertPreciseEqual(cfunc(d, 0), d)

        # sorted
        d = np.arange(49)
        for k in 5, 15:
            self.assertEqual(cfunc(d, k)[k], k)
            self.partition_sanity_check(pyfunc, cfunc, d, k)

        # rsorted, with input flavours: array, list and tuple
        d = np.arange(47)[::-1]
        for a in d, d.tolist(), tuple(d.tolist()):
            self.assertEqual(cfunc(a, 6)[6], 6)
            self.assertEqual(cfunc(a, 16)[16], 16)
            self.assertPreciseEqual(cfunc(a, -6), cfunc(a, 41))
            self.assertPreciseEqual(cfunc(a, -16), cfunc(a, 31))
            self.partition_sanity_check(pyfunc, cfunc, d, -16)

        # median of 3 killer, O(n^2) on pure median 3 pivot quickselect
        # exercises the median of median of 5 code used to keep O(n)
        d = np.arange(1000000)
        x = np.roll(d, d.size // 2)
        mid = x.size // 2 + 1
        self.assertEqual(cfunc(x, mid)[mid], mid)
        d = np.arange(1000001)
        x = np.roll(d, d.size // 2 + 1)
        mid = x.size // 2 + 1
        self.assertEqual(cfunc(x, mid)[mid], mid)

        # max
        d = np.ones(10)
        d[1] = 4
        self.assertEqual(cfunc(d, (2, -1))[-1], 4)
        self.assertEqual(cfunc(d, (2, -1))[2], 1)
        d[1] = np.nan
        assert np.isnan(cfunc(d, (2, -1))[-1])

        # equal elements
        d = np.arange(47) % 7
        tgt = np.sort(np.arange(47) % 7)
        self.rnd.shuffle(d)
        for i in range(d.size):
            self.assertEqual(cfunc(d, i)[i], tgt[i])
            self.partition_sanity_check(pyfunc, cfunc, d, i)

        d = np.array([0, 1, 2, 3, 4, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                      7, 7, 7, 7, 7, 9])
        kth = [0, 3, 19, 20]
        self.assertEqual(tuple(cfunc(d, kth)[kth]), (0, 3, 7, 7))

        td = [(dt, s) for dt in [np.int32, np.float32] for s in (9, 16)]
        for dt, s in td:
            d = np.arange(s, dtype=dt)
            self.rnd.shuffle(d)
            d1 = np.tile(np.arange(s, dtype=dt), (4, 1))
            map(self.rnd.shuffle, d1)
            for i in range(d.size):
                p = cfunc(d, i)
                self.assertEqual(p[i], i)
                # all before are smaller
                np.testing.assert_array_less(p[:i], p[i])
                # all after are larger
                np.testing.assert_array_less(p[i], p[i + 1:])
                # sanity check
                self.partition_sanity_check(pyfunc, cfunc, d, i)

    def assert_partitioned(self, pyfunc, cfunc, d, kth):
        prev = 0
        for k in np.sort(kth):
            np.testing.assert_array_less(d[prev:k], d[k], err_msg='kth %d' % k)
            assert (d[k:] >= d[k]).all(), "kth %d, %r not greater equal %d" % (k, d[k:], d[k])
            prev = k + 1
            self.partition_sanity_check(pyfunc, cfunc, d, k)

    def test_partition_iterative(self):
        # inspired by the test of the same name in:
        # https://github.com/numpy/numpy/blob/043a840/numpy/core/tests/test_multiarray.py
        pyfunc = partition
        cfunc = jit(nopython=True)(pyfunc)

        assert_partitioned = partial(self.assert_partitioned, pyfunc, cfunc)

        d = np.array([3, 4, 2, 1])
        p = cfunc(d, (0, 3))
        assert_partitioned(p, (0, 3))
        assert_partitioned(d[np.argpartition(d, (0, 3))], (0, 3))

        self.assertPreciseEqual(p, cfunc(d, (-3, -1)))

        d = np.arange(17)
        self.rnd.shuffle(d)
        self.assertPreciseEqual(np.arange(17), cfunc(d, list(range(d.size))))

        # test unsorted kth
        d = np.arange(17)
        self.rnd.shuffle(d)
        keys = np.array([1, 3, 8, -2])
        self.rnd.shuffle(d)
        p = cfunc(d, keys)
        assert_partitioned(p, keys)
        self.rnd.shuffle(keys)
        self.assertPreciseEqual(cfunc(d, keys), p)

        # equal kth
        d = np.arange(20)[::-1]
        assert_partitioned(cfunc(d, [5] * 4), [5])
        assert_partitioned(cfunc(d, [5] * 4 + [6, 13]), [5] * 4 + [6, 13])

    def test_partition_multi_dim(self):
        pyfunc = partition
        cfunc = jit(nopython=True)(pyfunc)

        def check(a, kth):
            expected = pyfunc(a, kth)
            got = cfunc(a, kth)
            self.assertPreciseEqual(expected[:, :, kth], got[:, :, kth])

            for s in np.ndindex(expected.shape[:-1]):
                self.assertPreciseEqual(np.unique(expected[s][:kth]), np.unique(got[s][:kth]))
                self.assertPreciseEqual(np.unique(expected[s][kth:]), np.unique(got[s][kth:]))

        def a_variations(a):
            yield a
            yield a.T
            yield np.asfortranarray(a)
            yield np.full_like(a, fill_value=np.nan)
            yield np.full_like(a, fill_value=np.inf)
            yield (((1.0, 3.142, -np.inf, 3),),)  # multi-dimensional tuple input

        a = np.linspace(1, 10, 48)
        a[4:7] = np.nan
        a[8] = -np.inf
        a[9] = np.inf
        a = a.reshape((4, 3, 4))

        for arr in a_variations(a):
            for k in range(-3, 3):
                check(arr, k)

    def test_partition_boolean_inputs(self):
        pyfunc = partition
        cfunc = jit(nopython=True)(pyfunc)

        for d in np.linspace(1, 10, 17), np.array((True, False, True)):
            for kth in True, False, -1, 0, 1:
                self.partition_sanity_check(pyfunc, cfunc, d, kth)

    @unittest.skipUnless(np_version >= (1, 10), "cov needs Numpy 1.10+")
    @needs_blas
    def test_cov_invalid_ddof(self):
        pyfunc = cov
        cfunc = jit(nopython=True)(pyfunc)

        # Exceptions leak references
        self.disable_leak_check()

        m = np.array([[0, 2], [1, 1], [2, 0]]).T

        for ddof in np.arange(4), 4j:
            with self.assertTypingError() as raises:
               cfunc(m, ddof=ddof)
            self.assertIn('ddof must be a real numerical scalar type', str(raises.exception))

        for ddof in np.nan, np.inf:
            with self.assertRaises(ValueError) as raises:
               cfunc(m, ddof=ddof)
            self.assertIn('Cannot convert non-finite ddof to integer', str(raises.exception))

        for ddof in 1.1, -0.7:
            with self.assertRaises(ValueError) as raises:
               cfunc(m, ddof=ddof)
            self.assertIn('ddof must be integral value', str(raises.exception))

    @unittest.skipUnless(np_version >= (1, 10), "cov needs Numpy 1.10+")
    @needs_blas
    def test_cov_basic(self):
        pyfunc = cov
        cfunc = jit(nopython=True)(pyfunc)
        _check = partial(self._check_output, pyfunc, cfunc, abs_tol=1e-14)

        def m_variations():
            # array inputs
            yield np.array([[0, 2], [1, 1], [2, 0]]).T
            yield self.rnd.randn(100).reshape(5, 20)
            yield np.asfortranarray(np.array([[0, 2], [1, 1], [2, 0]]).T)
            yield self.rnd.randn(100).reshape(5, 20)[:, ::2]
            yield np.array([0.3942, 0.5969, 0.7730, 0.9918, 0.7964])
            yield np.full((4, 5), fill_value=True)
            yield np.array([np.nan, 0.5969, -np.inf, 0.9918, 0.7964])
            yield np.linspace(-3, 3, 33).reshape(33, 1)

            # non-array inputs
            yield ((0.1, 0.2), (0.11, 0.19), (0.09, 0.21))  # UniTuple
            yield ((0.1, 0.2), (0.11, 0.19), (0.09j, 0.21j))  # Tuple
            yield (-2.1, -1, 4.3)
            yield (1, 2, 3)
            yield [4, 5, 6]
            yield ((0.1, 0.2, 0.3), (0.1, 0.2, 0.3))
            yield [(1, 2, 3), (1, 3, 2)]
            yield 3.142
            yield ((1.1, 2.2, 1.5),)

            # empty data structures
            yield np.array([])
            yield np.array([]).reshape(0, 2)
            yield np.array([]).reshape(2, 0)
            yield ()

        # all inputs other than the first are defaulted
        for m in m_variations():
            _check({'m': m})

    @unittest.skipUnless(np_version >= (1, 10), "cov needs Numpy 1.10+")
    @needs_blas
    def test_cov_explicit_arguments(self):
        pyfunc = cov
        cfunc = jit(nopython=True)(pyfunc)
        _check = partial(self._check_output, pyfunc, cfunc, abs_tol=1e-14)

        m = self.rnd.randn(1050).reshape(150, 7)
        y_choices = None, m[::-1]
        rowvar_choices = False, True
        bias_choices = False, True
        ddof_choice = None, -1, 0, 1, 3.0, True

        for y, rowvar, bias, ddof in itertools.product(y_choices, rowvar_choices, bias_choices, ddof_choice):
            params = {'m': m, 'y': y, 'ddof': ddof, 'bias': bias, 'rowvar': rowvar}
            _check(params)

    @unittest.skipUnless(np_version >= (1, 10), "cov needs Numpy 1.10+")
    @needs_blas
    def test_cov_edge_cases(self):
        pyfunc = cov
        cfunc = jit(nopython=True)(pyfunc)
        _check = partial(self._check_output, pyfunc, cfunc, abs_tol=1e-14)

        # some of these examples borrowed from numpy doc string examples:
        # https://github.com/numpy/numpy/blob/v1.15.0/numpy/lib/function_base.py#L2199-L2231
        # some borrowed from TestCov:
        # https://github.com/numpy/numpy/blob/80d3a7a/numpy/lib/tests/test_function_base.py
        m = np.array([-2.1, -1, 4.3])
        y = np.array([3, 1.1, 0.12])
        params = {'m': m, 'y': y}
        _check(params)

        m = np.array([[0, 2], [1, 1], [2, 0]]).T
        params = {'m': m, 'ddof': 5}
        _check(params)

        m = np.array([1, 2, 3])  # test case modified such that m is 1D
        y = np.array([[1j, 2j, 3j]])
        params = {'m': m, 'y': y}
        _check(params)

        m = np.array([1, 2, 3])
        y = (1j, 2j, 3j)
        params = {'m': m, 'y': y}
        _check(params)
        params = {'m': y, 'y': m}  # flip real and complex inputs
        _check(params)

        m = np.array([1, 2, 3])
        y = (1j, 2j, 3)  # note last item is not complex
        params = {'m': m, 'y': y}
        _check(params)
        params = {'m': y, 'y': m}  # flip real and complex inputs
        _check(params)

        m = np.array([])
        y = np.array([])
        params = {'m': m, 'y': y}
        _check(params)

        m = 1.1
        y = 2.2
        params = {'m': m, 'y': y}
        _check(params)

        m = self.rnd.randn(10, 3)
        y = np.array([-2.1, -1, 4.3]).reshape(1, 3) / 10
        params = {'m': m, 'y': y}
        _check(params)

        # The following tests pass with numpy version >= 1.10, but fail with 1.9
        m = np.array([-2.1, -1, 4.3])
        y = np.array([[3, 1.1, 0.12], [3, 1.1, 0.12]])
        params = {'m': m, 'y': y}
        _check(params)

        for rowvar in False, True:
            m = np.array([-2.1, -1, 4.3])
            y = np.array([[3, 1.1, 0.12], [3, 1.1, 0.12], [4, 1.1, 0.12]])
            params = {'m': m, 'y': y, 'rowvar': rowvar}
            _check(params)
            params = {'m': y, 'y': m, 'rowvar': rowvar}  # swap m and y
            _check(params)

    @unittest.skipUnless(np_version >= (1, 10), "cov needs Numpy 1.10+")
    @needs_blas
    def test_cov_exceptions(self):
        pyfunc = cov
        cfunc = jit(nopython=True)(pyfunc)

        # Exceptions leak references
        self.disable_leak_check()

        def _check_m(m):
            with self.assertTypingError() as raises:
                cfunc(m)
            self.assertIn('m has more than 2 dimensions', str(raises.exception))

        m = np.ones((5, 6, 7))
        _check_m(m)

        m = ((((1, 2, 3), (2, 2, 2)),),)
        _check_m(m)

        m = [[[5, 6, 7]]]
        _check_m(m)

        def _check_y(m, y):
            with self.assertTypingError() as raises:
                cfunc(m, y=y)
            self.assertIn('y has more than 2 dimensions', str(raises.exception))

        m = np.ones((5, 6))
        y = np.ones((5, 6, 7))
        _check_y(m, y)

        m = np.array((1.1, 2.2, 1.1))
        y = (((1.2, 2.2, 2.3),),)
        _check_y(m, y)

        m = np.arange(3)
        y = np.arange(4)
        with self.assertRaises(ValueError) as raises:
            cfunc(m, y=y)
        self.assertIn('m and y have incompatible dimensions', str(raises.exception))
        # Numpy raises ValueError: all the input array dimensions except for the
        # concatenation axis must match exactly.

        m = np.array([-2.1, -1, 4.3]).reshape(1, 3)
        with self.assertRaises(RuntimeError) as raises:
            cfunc(m)
        self.assertIn('2D array containing a single row is unsupported', str(raises.exception))

    @unittest.skipUnless(np_version >= (1, 12), "ediff1d needs Numpy 1.12+")
    def test_ediff1d_basic(self):
        pyfunc = ediff1d
        cfunc = jit(nopython=True)(pyfunc)
        _check = partial(self._check_output, pyfunc, cfunc)

        def to_variations(a):
            yield None
            yield a
            yield a.astype(np.int16)

        def ary_variations(a):
            yield a
            yield a.reshape(3, 2, 2)
            yield a.astype(np.int32)

        for ary in ary_variations(np.linspace(-2, 7, 12)):
            params = {'ary': ary}
            _check(params)

            for a in to_variations(ary):
                params = {'ary': ary, 'to_begin': a}
                _check(params)

                params = {'ary': ary, 'to_end': a}
                _check(params)

                for b in to_variations(ary):
                    params = {'ary': ary, 'to_begin': a, 'to_end': b}
                    _check(params)

    @unittest.skipUnless(np_version >= (1, 12), "ediff1d needs Numpy 1.12+")
    def test_ediff1d_edge_cases(self):
        pyfunc = ediff1d
        cfunc = jit(nopython=True)(pyfunc)
        _check = partial(self._check_output, pyfunc, cfunc)

        def input_variations():
            yield ((1, 2, 3), (4, 5, 6))
            yield [4, 5, 6]
            yield np.array([])
            yield ()
            yield np.array([np.nan, np.inf, 4, -np.inf, 3.142])
            parts = np.array([np.nan, 2, np.nan, 4, 5, 6, 7, 8, 9])
            a = parts + 1j * parts[::-1]
            yield a.reshape(3, 3)

        for i in input_variations():
            params = {'ary': i, 'to_end': i, 'to_begin': i}
            _check(params)

        # to_end / to_begin are boolean
        params = {'ary': [1], 'to_end': (False,), 'to_begin': (True, False)}
        _check(params)

        # example of unsafe type casting (np.nan to np.int32)
        to_begin = np.array([1, 2, 3.142, np.nan, 5, 6, 7, -8, np.nan])
        params = {'ary': np.arange(-4, 6), 'to_begin': to_begin}
        _check(params)

        # scalar inputs
        params = {'ary': 3.142}
        _check(params)

        params = {'ary': 3, 'to_begin': 3.142}
        _check(params)

        params = {'ary': np.arange(-4, 6), 'to_begin': -5, 'to_end': False}
        _check(params)

        # the following would fail on one of the BITS32 builds (difference in
        # overflow handling):
        # params = {'ary': np.array([5, 6], dtype=np.int16), 'to_end': [1e100]}
        # _check(params)

    @unittest.skipUnless(np_version >= (1, 12), "ediff1d needs Numpy 1.12+")
    def test_ediff1d_exceptions(self):
        pyfunc = ediff1d
        cfunc = jit(nopython=True)(pyfunc)

        # Exceptions leak references
        self.disable_leak_check()

        with self.assertTypingError() as e:
            cfunc(np.array((True, True, False)))

        msg = "Boolean dtype is unsupported (as per NumPy)"
        assert msg in str(e.exception)


class TestNPMachineParameters(TestCase):
    # tests np.finfo, np.iinfo, np.MachAr

    template = '''
def foo():
    ty = np.%s
    return np.%s(ty)
'''

    bits = ('bits',) if np_version >= (1, 12)  else ()

    def check(self, func, attrs, *args):
        pyfunc = func
        cfunc = jit(nopython=True)(pyfunc)

        expected = pyfunc(*args)
        got = cfunc(*args)

        # check result
        for attr in attrs:
            self.assertPreciseEqual(getattr(expected, attr),
                                    getattr(got, attr))

    def create_harcoded_variant(self, basefunc, ty):
        #create an instance of using the function with a hardcoded type
        #and eval it into existence, return the function for use
        tystr = ty.__name__
        basestr = basefunc.__name__
        funcstr = self.template % (tystr, basestr)
        eval(compile(funcstr, '<string>', 'exec'))
        return locals()['foo']

    def test_MachAr(self):
        attrs = ('ibeta', 'it', 'machep', 'eps', 'negep', 'epsneg', 'iexp',
                 'minexp', 'xmin', 'maxexp', 'xmax', 'irnd', 'ngrd',
                 'epsilon', 'tiny', 'huge', 'precision', 'resolution',)
        self.check(machar, attrs)

    def test_finfo(self):
        types = [np.float32, np.float64, np.complex64, np.complex128]
        attrs = self.bits + ('eps', 'epsneg', 'iexp', 'machep', 'max',
                'maxexp', 'negep', 'nexp', 'nmant', 'precision',
                'resolution', 'tiny',)
        for ty in types:
            self.check(finfo, attrs, ty(1))
            hc_func = self.create_harcoded_variant(np.finfo, ty)
            self.check(hc_func, attrs)

        # check unsupported attr raises
        with self.assertRaises(TypingError) as raises:
            cfunc = jit(nopython=True)(finfo_machar)
            cfunc(7.)
        msg = "Unknown attribute 'machar' of type finfo"
        self.assertIn(msg, str(raises.exception))

        # check invalid type raises
        with self.assertTypingError():
            cfunc = jit(nopython=True)(finfo)
            cfunc(np.int32(7))

    def test_iinfo(self):
        # check types and instances of types
        types = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,
                 np.uint32, np.uint64]
        attrs = ('min', 'max') + self.bits
        for ty in types:
            self.check(iinfo, attrs, ty(1))
            hc_func = self.create_harcoded_variant(np.iinfo, ty)
            self.check(hc_func, attrs)

        # check invalid type raises
        with self.assertTypingError():
            cfunc = jit(nopython=True)(iinfo)
            cfunc(np.float64(7))
