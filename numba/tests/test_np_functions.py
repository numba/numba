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
from numba.errors import TypingError
from .support import TestCase, CompilationCache, MemoryLeakMixin

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

def tri(N, M=None, k=0):
    return np.tri(N, M, k)

def tril(m, k=0):
    return np.tril(m, k)

def triu(m, k=0):
    return np.triu(m, k)

def np_vander(x, N=None, increasing=False):
    return np.vander(x, N, increasing)


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

    def test_vander_basic(self):
        pyfunc = np_vander
        cfunc = jit(nopython=True)(pyfunc)

        def _check_output(params):
            expected = pyfunc(**params)
            got = cfunc(**params)
            self.assertPreciseEqual(expected, got)

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
        pyfunc = np_vander
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

    def test_tri_basic(self):
        pyfunc = tri
        cfunc = jit(nopython=True)(pyfunc)

        def n_variations():
            return np.arange(-4, 8)  # number of rows

        def m_variations():
            return itertools.chain.from_iterable(([None], range(-5, 9)))  # number of columns

        def k_variations():
            return np.arange(-10, 10)  # offset

        def _check(params):
            expected = pyfunc(**params)
            got = cfunc(**params)
            self.assertPreciseEqual(expected, got)

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
        pyfunc = tri
        cfunc = jit(nopython=True)(pyfunc)

        # Exceptions leak references
        self.disable_leak_check()

        def _check(k):
            with self.assertTypingError() as raises:
                cfunc(5, 6, k=k)
            assert "k must be an integer" in str(raises.exception)

        for k in 1.5, True, np.inf, [1, 2]:
            _check(k)

    def _triangular_matrix_tests(self, pyfunc):
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
        self._triangular_matrix_tests(tril)

    def test_tril_exceptions(self):
        self._triangular_matrix_exceptions(tril)

    def test_triu_basic(self):
        self._triangular_matrix_tests(triu)

    def test_triu_exceptions(self):
        self._triangular_matrix_exceptions(triu)


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
