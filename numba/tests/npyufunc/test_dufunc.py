import itertools
import pickle
import textwrap

import numpy as np

from numba import njit, vectorize
from numba.tests.support import MemoryLeakMixin, TestCase
from numba.core.errors import TypingError
import unittest
from numba.np.ufunc import dufunc


def pyuadd(a0, a1):
    return a0 + a1


def pysub(a0, a1):
    return a0 - a1


def pymult(a0, a1):
    return a0 * a1


def pydiv(a0, a1):
    return a0 // a1


def pymin(a0, a1):
    return a0 if a0 < a1 else a1


class TestDUFunc(MemoryLeakMixin, unittest.TestCase):

    def nopython_dufunc(self, pyfunc):
        return dufunc.DUFunc(pyfunc, targetoptions=dict(nopython=True))

    def test_frozen(self):
        duadd = self.nopython_dufunc(pyuadd)
        self.assertFalse(duadd._frozen)
        duadd._frozen = True
        self.assertTrue(duadd._frozen)
        with self.assertRaises(ValueError):
            duadd._frozen = False
        with self.assertRaises(TypeError):
            duadd(np.linspace(0,1,10), np.linspace(1,2,10))

    def test_scalar(self):
        duadd = self.nopython_dufunc(pyuadd)
        self.assertEqual(pyuadd(1,2), duadd(1,2))

    def test_npm_call(self):
        duadd = self.nopython_dufunc(pyuadd)

        @njit
        def npmadd(a0, a1, o0):
            duadd(a0, a1, o0)
        X = np.linspace(0,1.9,20)
        X0 = X[:10]
        X1 = X[10:]
        out0 = np.zeros(10)
        npmadd(X0, X1, out0)
        np.testing.assert_array_equal(X0 + X1, out0)
        Y0 = X0.reshape((2,5))
        Y1 = X1.reshape((2,5))
        out1 = np.zeros((2,5))
        npmadd(Y0, Y1, out1)
        np.testing.assert_array_equal(Y0 + Y1, out1)
        Y2 = X1[:5]
        out2 = np.zeros((2,5))
        npmadd(Y0, Y2, out2)
        np.testing.assert_array_equal(Y0 + Y2, out2)

    def test_npm_call_implicit_output(self):
        duadd = self.nopython_dufunc(pyuadd)

        @njit
        def npmadd(a0, a1):
            return duadd(a0, a1)
        X = np.linspace(0,1.9,20)
        X0 = X[:10]
        X1 = X[10:]
        out0 = npmadd(X0, X1)
        np.testing.assert_array_equal(X0 + X1, out0)
        Y0 = X0.reshape((2,5))
        Y1 = X1.reshape((2,5))
        out1 = npmadd(Y0, Y1)
        np.testing.assert_array_equal(Y0 + Y1, out1)
        Y2 = X1[:5]
        out2 = npmadd(Y0, Y2)
        np.testing.assert_array_equal(Y0 + Y2, out2)
        out3 = npmadd(1.,2.)
        self.assertEqual(out3, 3.)

    def test_ufunc_props(self):
        duadd = self.nopython_dufunc(pyuadd)
        self.assertEqual(duadd.nin, 2)
        self.assertEqual(duadd.nout, 1)
        self.assertEqual(duadd.nargs, duadd.nin + duadd.nout)
        self.assertEqual(duadd.ntypes, 0)
        self.assertEqual(duadd.types, [])
        self.assertEqual(duadd.identity, None)
        duadd(1, 2)
        self.assertEqual(duadd.ntypes, 1)
        self.assertEqual(duadd.ntypes, len(duadd.types))
        self.assertIsNone(duadd.signature)

    def test_ufunc_props_jit(self):
        duadd = self.nopython_dufunc(pyuadd)
        duadd(1, 2)  # initialize types attribute

        attributes = {'nin': duadd.nin,
                      'nout': duadd.nout,
                      'nargs': duadd.nargs,
                      #'ntypes': duadd.ntypes,
                      #'types': duadd.types,
                      'identity': duadd.identity,
                      'signature': duadd.signature}

        def get_attr_fn(attr):
            fn = f'''
                def impl():
                    return duadd.{attr}
            '''
            l = {}
            exec(textwrap.dedent(fn), {'duadd': duadd}, l)
            return l['impl']

        for attr, val in attributes.items():
            cfunc = njit(get_attr_fn(attr))
            self.assertEqual(val, cfunc(),
                             f'Attribute differs from original: {attr}')

        # We don't expose [n]types attributes as they are dynamic attributes
        # and can change as the user calls the ufunc
        # cfunc = njit(get_attr_fn('ntypes'))
        # self.assertEqual(cfunc(), 1)
        # duadd(1.1, 2.2)
        # self.assertEqual(cfunc(), 2)


class TestDUFuncReduceNumPyTests(TestCase):
    # Tests taken from
    # https://github.com/numpy/numpy/blob/51ee17b6bd4ccec60a5483ee8bff94ad0c0e8585/numpy/_core/tests/test_ufunc.py  # noqa: E501

    def _generate_jit(self, ufunc, identity=None):
        if ufunc.nin == 2:
            vec = vectorize(identity=identity)(lambda a, b: ufunc(a, b))
        else:
            vec = vectorize(identity=identity)(lambda a: ufunc(a))

        @njit
        def fn(array, axis=0, initial=None):
            return vec.reduce(array, axis=axis, initial=initial)
        return fn

    @unittest.expectedFailure
    def test_numpy_scalar_reduction(self):
        # scalar reduction is not supported
        power_reduce = self._generate_jit(np.power)
        expected = np.power.reduce(3)
        got = power_reduce(3)
        self.assertPreciseEqual(expected, got)

    def check_identityless_reduction(self, a):
        def compare_output(a, b):
            # We don't use self.assertPreciseEqual as the dtype differs
            # between the value from the reduction and the hardcoded output
            np.testing.assert_equal(a, b)
        # test taken from:
        # https://github.com/numpy/numpy/blob/51ee17b6bd4ccec60a5483ee8bff94ad0c0e8585/numpy/_core/tests/test_ufunc.py#L1591  # noqa: E501

        minimum_reduce = self._generate_jit(np.minimum, identity='reorderable')

        # np.minimum.reduce is an identityless reduction

        # Verify that it sees the zero at various positions
        a[...] = 1
        a[1, 0, 0] = 0
        compare_output(minimum_reduce(a, axis=None), 0)
        compare_output(minimum_reduce(a, axis=(0, 1)), [0, 1, 1, 1])
        compare_output(minimum_reduce(a, axis=(0, 2)), [0, 1, 1])
        compare_output(minimum_reduce(a, axis=(1, 2)), [1, 0])
        compare_output(minimum_reduce(a, axis=0),
                       [[0, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
        compare_output(minimum_reduce(a, axis=1),
                       [[1, 1, 1, 1], [0, 1, 1, 1]])
        compare_output(minimum_reduce(a, axis=2),
                       [[1, 1, 1], [0, 1, 1]])
        compare_output(minimum_reduce(a, axis=()), a)

        a[...] = 1
        a[0, 1, 0] = 0
        compare_output(minimum_reduce(a, axis=None), 0)
        compare_output(minimum_reduce(a, axis=(0, 1)), [0, 1, 1, 1])
        compare_output(minimum_reduce(a, axis=(0, 2)), [1, 0, 1])
        compare_output(minimum_reduce(a, axis=(1, 2)), [0, 1])
        compare_output(minimum_reduce(a, axis=0),
                       [[1, 1, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]])
        compare_output(minimum_reduce(a, axis=1),
                       [[0, 1, 1, 1], [1, 1, 1, 1]])
        compare_output(minimum_reduce(a, axis=2),
                       [[1, 0, 1], [1, 1, 1]])
        compare_output(minimum_reduce(a, axis=()), a)

        a[...] = 1
        a[0, 0, 1] = 0
        compare_output(minimum_reduce(a, axis=None), 0)
        compare_output(minimum_reduce(a, axis=(0, 1)), [1, 0, 1, 1])
        compare_output(minimum_reduce(a, axis=(0, 2)), [0, 1, 1])
        compare_output(minimum_reduce(a, axis=(1, 2)), [0, 1])
        compare_output(minimum_reduce(a, axis=0),
                       [[1, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
        compare_output(minimum_reduce(a, axis=1),
                       [[1, 0, 1, 1], [1, 1, 1, 1]])
        compare_output(minimum_reduce(a, axis=2),
                       [[0, 1, 1], [1, 1, 1]])
        compare_output(minimum_reduce(a, axis=()), a)

    def test_numpy_identityless_reduction_corder(self):
        a = np.empty((2, 3, 4), order='C')
        self.check_identityless_reduction(a)

    def test_numpy_identityless_reduction_forder(self):
        a = np.empty((2, 3, 4), order='F')
        self.check_identityless_reduction(a)

    def test_numpy_identityless_reduction_otherorder(self):
        a = np.empty((2, 4, 3), order='C').swapaxes(1, 2)
        self.check_identityless_reduction(a)

    def test_numpy_identityless_reduction_noncontig(self):
        a = np.empty((3, 5, 4), order='C').swapaxes(1, 2)
        a = a[1:, 1:, 1:]
        self.check_identityless_reduction(a)

    def test_numpy_identityless_reduction_noncontig_unaligned(self):
        a = np.empty((3 * 4 * 5 * 8 + 1,), dtype='i1')
        a = a[1:].view(dtype='f8')
        a.shape = (3, 4, 5)
        a = a[1:, 1:, 1:]
        self.check_identityless_reduction(a)

    def test_numpy_initial_reduction(self):
        # np.minimum.reduce is an identityless reduction
        add_reduce = self._generate_jit(np.add)
        min_reduce = self._generate_jit(np.minimum)
        max_reduce = self._generate_jit(np.maximum)

        # For cases like np.maximum(np.abs(...), initial=0)
        # More generally, a supremum over non-negative numbers.
        self.assertPreciseEqual(max_reduce(np.asarray([]), initial=0), 0.0)

        # For cases like reduction of an empty array over the reals.
        self.assertPreciseEqual(min_reduce(np.asarray([]), initial=np.inf),
                                np.inf)
        self.assertPreciseEqual(max_reduce(np.asarray([]), initial=-np.inf),
                                -np.inf)

        # Random tests
        self.assertPreciseEqual(min_reduce(np.asarray([5]), initial=4), 4)
        self.assertPreciseEqual(max_reduce(np.asarray([4]), initial=5), 5)
        self.assertPreciseEqual(max_reduce(np.asarray([5]), initial=4), 5)
        self.assertPreciseEqual(min_reduce(np.asarray([4]), initial=5), 4)

        # Check initial=None raises ValueError for both types of ufunc
        # reductions
        msg = 'zero-size array to reduction operation'
        for func in (add_reduce, min_reduce):
            with self.assertRaisesRegex(ValueError, msg):
                func(np.asarray([]), initial=None)

    def test_numpy_empty_reduction_and_identity(self):
        arr = np.zeros((0, 5))
        true_divide_reduce = self._generate_jit(np.true_divide)

        # OK, since the reduction itself is *not* empty, the result is
        expected = np.true_divide.reduce(arr, axis=1)
        got = true_divide_reduce(arr, axis=1)
        self.assertPreciseEqual(expected, got)
        self.assertPreciseEqual(got.shape, (0,))

        # Not OK, the reduction itself is empty and we have no identity
        msg = 'zero-size array to reduction operation'
        with self.assertRaisesRegex(ValueError, msg):
            true_divide_reduce(arr, axis=0)

        # Test that an empty reduction fails also if the result is empty
        arr = np.zeros((0, 0, 5))
        with self.assertRaisesRegex(ValueError, msg):
            true_divide_reduce(arr, axis=1)

        # Division reduction makes sense with `initial=1` (empty or not):
        expected = np.true_divide.reduce(arr, axis=1, initial=1)
        got = true_divide_reduce(arr, axis=1, initial=1)
        self.assertPreciseEqual(expected, got)

    def test_identityless_reduction_nonreorderable(self):
        a = np.array([[8.0, 2.0, 2.0], [1.0, 0.5, 0.25]])

        divide_reduce = self._generate_jit(np.divide)
        res = divide_reduce(a, axis=0)
        self.assertPreciseEqual(res, np.asarray([8.0, 4.0, 8.0]))

        res = divide_reduce(a, axis=1)
        self.assertPreciseEqual(res, np.asarray([2.0, 8.0]))

        res = divide_reduce(a, axis=())
        self.assertPreciseEqual(res, a)

        # will not raise as per Numba issue #9283
        # assert_raises(ValueError, np.divide.reduce, a, axis=(0, 1))

    def test_reduce_zero_axis(self):
        # If we have a n x m array and do a reduction with axis=1, then we are
        # doing n reductions, and each reduction takes an m-element array. For
        # a reduction operation without an identity, then:
        #   n > 0, m > 0: fine
        #   n = 0, m > 0: fine, doing 0 reductions of m-element arrays
        #   n > 0, m = 0: can't reduce a 0-element array, ValueError
        #   n = 0, m = 0: can't reduce a 0-element array, ValueError (for
        #     consistency with the above case)
        # This test doesn't actually look at return values, it just checks to
        # make sure that error we get an error in exactly those cases where we
        # expect one, and assumes the calculations themselves are done
        # correctly.

        def ok(f, *args, **kwargs):
            f(*args, **kwargs)

        def err(f, *args, **kwargs):
            with self.assertRaises(ValueError):
                f(*args, **kwargs)

        def t(expect, func, n, m):
            expect(func, np.zeros((n, m)), axis=1)
            expect(func, np.zeros((m, n)), axis=0)
            expect(func, np.zeros((n // 2, n // 2, m)), axis=2)
            expect(func, np.zeros((n // 2, m, n // 2)), axis=1)
            expect(func, np.zeros((n, m // 2, m // 2)), axis=(1, 2))
            expect(func, np.zeros((m // 2, n, m // 2)), axis=(0, 2))
            expect(func, np.zeros((m // 3, m // 3, m // 3,
                                   n // 2, n // 2)), axis=(0, 1, 2))
            # Check what happens if the inner (resp. outer) dimensions are a
            # mix of zero and non-zero:
            expect(func, np.zeros((10, m, n)), axis=(0, 1))
            expect(func, np.zeros((10, n, m)), axis=(0, 2))
            expect(func, np.zeros((m, 10, n)), axis=0)
            expect(func, np.zeros((10, m, n)), axis=1)
            expect(func, np.zeros((10, n, m)), axis=2)

        # np.maximum is just an arbitrary ufunc with no reduction identity
        maximum_reduce = self._generate_jit(np.maximum, identity='reorderable')
        self.assertEqual(np.maximum.identity, None)
        t(ok, maximum_reduce, 30, 30)
        t(ok, maximum_reduce, 0, 30)
        t(err, maximum_reduce, 30, 0)
        t(err, maximum_reduce, 0, 0)
        err(maximum_reduce, [])
        maximum_reduce(np.zeros((0, 0)), axis=())

        # all of the combinations are fine for a reduction that has an
        # identity
        add_reduce = self._generate_jit(np.add, identity=0)
        t(ok, add_reduce, 30, 30)
        t(ok, add_reduce, 0, 30)
        t(ok, add_reduce, 30, 0)
        t(ok, add_reduce, 0, 0)
        add_reduce(np.array([], dtype=np.int64))
        add_reduce(np.zeros((0, 0)), axis=())


class TestDUFuncMethods(TestCase):
    def _check_reduce(self, ufunc, dtype=None, initial=None):

        @njit
        def foo(a, axis, dtype, initial):
            return ufunc.reduce(a,
                                axis=axis,
                                dtype=dtype,
                                initial=initial)

        inputs = [
            np.arange(5),
            np.arange(4).reshape(2, 2),
            np.arange(40).reshape(5, 4, 2),
        ]
        for array in inputs:
            for axis in range(array.ndim):
                expected = foo.py_func(array, axis, dtype, initial)
                got = foo(array, axis, dtype, initial)
                self.assertPreciseEqual(expected, got)

    def _check_reduce_axis(self, ufunc, dtype, initial=None):

        @njit
        def foo(a, axis):
            return ufunc.reduce(a, axis=axis, initial=initial)

        def _check(*args):
            try:
                expected = foo.py_func(array, axis)
            except ValueError as e:
                self.assertEqual(e.args[0], exc_msg)
                with self.assertRaisesRegex(TypingError, exc_msg):
                    got = foo(array, axis)
            else:
                got = foo(array, axis)
                self.assertPreciseEqual(expected, got)

        exc_msg = (f"reduction operation '{ufunc.__name__}' is not "
                   "reorderable, so at most one axis may be specified")
        inputs = [
            np.arange(40, dtype=dtype).reshape(5, 4, 2),
            np.arange(10, dtype=dtype),
        ]
        for array in inputs:
            for i in range(1, array.ndim + 1):
                for axis in itertools.combinations(range(array.ndim), r=i):
                    _check(array, axis)

            # corner cases: Reduce over axis=() and axis=None
            for axis in ((), None):
                _check(array, axis)

    def test_add_reduce(self):
        duadd = vectorize('int64(int64, int64)', identity=0)(pyuadd)
        self._check_reduce(duadd)
        self._check_reduce_axis(duadd, dtype=np.int64)

    def test_mul_reduce(self):
        dumul = vectorize('int64(int64, int64)', identity=1)(pymult)
        self._check_reduce(dumul)

    def test_non_associative_reduce(self):
        dusub = vectorize('int64(int64, int64)', identity=None)(pysub)
        dudiv = vectorize('int64(int64, int64)', identity=None)(pydiv)
        self._check_reduce(dusub)
        self._check_reduce_axis(dusub, dtype=np.int64)
        self._check_reduce(dudiv)
        self._check_reduce_axis(dudiv, dtype=np.int64)

    def test_reduce_dtype(self):
        duadd = vectorize('float64(float64, int64)', identity=0)(pyuadd)
        self._check_reduce(duadd, dtype=np.float64)

    def test_min_reduce(self):
        dumin = vectorize('int64(int64, int64)', identity='reorderable')(pymin)
        self._check_reduce(dumin, initial=10)
        self._check_reduce_axis(dumin, dtype=np.int64)

    def test_add_reduce_initial(self):
        # Initial should be used as a start
        duadd = vectorize('int64(int64, int64)', identity=0)(pyuadd)
        self._check_reduce(duadd, dtype=np.int64, initial=100)

    def test_add_reduce_no_initial_or_identity(self):
        # don't provide an initial or identity value
        duadd = vectorize('int64(int64, int64)')(pyuadd)
        self._check_reduce(duadd, dtype=np.int64)

    def test_invalid_input(self):
        duadd = vectorize('float64(float64, int64)', identity=0)(pyuadd)

        @njit
        def foo(a):
            return duadd.reduce(a)

        exc_msg = 'The first argument "array" must be array-like'
        with self.assertRaisesRegex(TypingError, exc_msg):
            foo('a')

    def test_dufunc_negative_axis(self):
        duadd = vectorize('int64(int64, int64)', identity=0)(pyuadd)

        @njit
        def foo(a, axis):
            return duadd.reduce(a, axis=axis)

        a = np.arange(40).reshape(5, 4, 2)
        cases = (0, -1, (0, -1), (-1, -2), (1, -1), -3)
        for axis in cases:
            expected = duadd.reduce(a, axis)
            got = foo(a, axis)
            self.assertPreciseEqual(expected, got)

    def test_dufunc_invalid_axis(self):
        duadd = vectorize('int64(int64, int64)', identity=0)(pyuadd)

        @njit
        def foo(a, axis):
            return duadd.reduce(a, axis=axis)

        a = np.arange(40).reshape(5, 4, 2)
        cases = ((0, 0), (0, 1, 0), (0, -3), (-1, -1), (-1, 2))
        for axis in cases:
            msg = "duplicate value in 'axis'"
            with self.assertRaisesRegex(ValueError, msg):
                foo(a, axis)

        cases = (-4, 3, (0, -4),)
        for axis in cases:
            with self.assertRaisesRegex(ValueError, "Invalid axis"):
                foo(a, axis)


class TestDUFuncPickling(MemoryLeakMixin, unittest.TestCase):
    def check(self, ident, result_type):
        buf = pickle.dumps(ident)
        rebuilt = pickle.loads(buf)

        # Check reconstructed dufunc
        r = rebuilt(123)
        self.assertEqual(123, r)
        self.assertIsInstance(r, result_type)

        # Try to use reconstructed dufunc in @jit
        @njit
        def foo(x):
            return rebuilt(x)

        r = foo(321)
        self.assertEqual(321, r)
        self.assertIsInstance(r, result_type)

    def test_unrestricted(self):
        @vectorize
        def ident(x1):
            return x1

        self.check(ident, result_type=(int, np.integer))

    def test_restricted(self):
        @vectorize(["float64(float64)"])
        def ident(x1):
            return x1

        self.check(ident, result_type=float)


if __name__ == "__main__":
    unittest.main()
