import functools
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


class TestDUFuncMethodsBase(TestCase):

    @functools.cache
    def _generate_jit(self, ufunc, kind, identity=None):
        assert kind in ('reduce', 'reduceat', 'at')
        if kind == 'reduce':
            if ufunc.nin == 2:
                vec = vectorize(identity=identity)(lambda a, b: ufunc(a, b))
            else:
                vec = vectorize(identity=identity)(lambda a: ufunc(a))

            @njit
            def fn(array, axis=0, initial=None):
                return vec.reduce(array, axis=axis, initial=initial)
            return fn
        elif kind == 'reduceat':
            if ufunc.nin != 2:
                raise ValueError('reduceat only supported for binary functions')
            vec = vectorize(identity=identity)(lambda a, b: ufunc(a, b))

            @njit
            def fn(array, indices, axis=0, dtype=None, out=None):
                return vec.reduceat(array, indices, axis, dtype, out)
            return fn
        else:
            if ufunc.nin == 2:
                vec = vectorize(identity=identity)(lambda a, b: ufunc(a, b))
            else:
                vec = vectorize(identity=identity)(lambda a: ufunc(a))

            @njit
            def fn(*args):
                return vec.at(*args)
            return fn

    def _reduce(self, ufunc, identity):
        return self._generate_jit(ufunc, 'reduce', identity=identity)

    def _reduceat(self, ufunc):
        return self._generate_jit(ufunc, 'reduceat')

    def _at(self, ufunc):
        return self._generate_jit(ufunc, 'at')


class TestDUFuncMethods(TestDUFuncMethodsBase):
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
        dusub = vectorize('int64(int64, int64)')(pysub)
        dudiv = vectorize('int64(int64, int64)')(pydiv)
        self._check_reduce(dusub)
        self._check_reduce_axis(dusub, dtype=np.int64)
        self._check_reduce(dudiv)
        self._check_reduce_axis(dudiv, dtype=np.int64)

    def test_reduce_dtype(self):
        duadd = vectorize('float64(float64, int64)', identity=0)(pyuadd)
        self._check_reduce(duadd, dtype=np.float64)

    def test_min_reduce(self):
        dumin = vectorize('int64(int64, int64)')(pymin)
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


class TestDUFuncReduceAt(TestDUFuncMethodsBase):
    def _compare_output(self, ufunc, a, idx, **kwargs):
        fn = self._reduceat(ufunc)
        expected = a.copy()
        got = a.copy()
        ufunc.reduceat(expected, idx, **kwargs)
        fn(got, idx, **kwargs)
        self.assertPreciseEqual(expected, got)

    def test_reduceat_out_kw(self):
        arr = np.arange(4)
        idx = np.asarray([0, 3, 1, 2])
        add_reduce = self._reduceat(np.add)
        add_reduce(arr, idx, out=arr)
        self.assertPreciseEqual(np.asarray([3, 3, 3, 6]), arr)
        add_reduce(arr, idx, out=arr)
        self.assertPreciseEqual(np.asarray([9, 6, 6, 12]), arr)

    @unittest.expectedFailure
    def test_reduceat_axis_kw(self):
        arrays = (
            np.arange(16).reshape(4, 4),
            np.arange(40).reshape(4, 5, 2),
            np.ones((4, 4))
        )
        indices = (
            np.asarray([0, 3, 1, 2, 0]),
            np.asarray([0, 3, 1, 2]),
        )
        axis = (1, 0, -1)  # needs gh-9296
        for array in arrays:
            for idx in indices:
                for ax in axis:
                    self._compare_output(np.add, array, idx, axis=ax)

    @unittest.expectedFailure
    def test_reduceat_invalid_axis(self):
        arr = np.ones((4, 4))
        idx = np.asarray([0, 3, 1, 2])
        add_reduceat = self._reduceat(np.add)

        for ax in (2, -2):  # needs gh-9296
            msg = (f'axis {ax} is out of bounds for array of dimension '
                   f'{arr.ndim}')
            with self.assertRaisesRegex(ValueError, msg):
                add_reduceat(arr, idx, ax)

    def test_reduceat_cast_args_to_array(self):
        add_reduceat = self._reduceat(np.add)

        # cast array and indices
        a = [1, 2, 3, 4]
        idx = [1, 2, 3]
        expected = np.add.reduceat(a, idx)
        got = add_reduceat(a, idx)
        self.assertPreciseEqual(expected, got)

        # array and indices as tuples
        a = (1, 2, 3, 4)
        idx = (1, 2, 3)
        expected = np.add.reduceat(a, idx)
        got = add_reduceat(a, idx)
        self.assertPreciseEqual(expected, got)

    # tests below this line were copied from NumPy
    # https://github.com/numpy/numpy/blob/7f8dc13b9bcaa26cb378b9c8246110ca1dc9ce75/numpy/_core/tests/test_ufunc.py#L200C1-L200C1  # noqa: E501
    def test_reduceat_basic(self):
        x = np.arange(8)
        idx = [0,4, 1,5, 2,6, 3,7]
        self._compare_output(np.add, x, idx)

    def test_reduceat_basic_2d(self):
        x = np.linspace(0, 15, 16).reshape(4, 4)
        idx = [0, 3, 1, 2, 0]
        self._compare_output(np.add, x, idx)

    def test_reduceat_shifting_sum(self):
        L = 6
        x = np.arange(L)
        idx = np.array(list(zip(np.arange(L - 2), np.arange(L - 2) + 2))).ravel()  # noqa: E501
        self._compare_output(np.add, x, idx)

    @unittest.expectedFailure
    def test_reduceat_int_array_reduceat_inplace(self):
        # Checks that in-place reduceats work, see also gh-7465
        arr = np.ones(4, dtype=np.int64)
        out = np.ones(4, dtype=np.int64)
        self._compare_output(np.add, arr, np.arange(4), out=arr)
        self._compare_output(np.add, arr, np.arange(4), out=arr)
        self.assertPreciseEqual(arr, out)

        # needs gh-9296
        # And the same if the axis argument is used
        arr = np.ones((2, 4), dtype=np.int64)
        arr[0, :] = [2 for i in range(4)]
        out = np.ones((2, 4), dtype=np.int64)
        out[0, :] = [2 for i in range(4)]
        self._compare_output(np.add, arr, np.arange(4), out=arr, axis=-1)
        self._compare_output(np.add, arr, np.arange(4), out=arr, axis=-1)
        self.assertPreciseEqual(arr, out)

    def test_reduceat_out_shape_mismatch(self):
        # Should raise an error mentioning "shape" or "size"
        # original test has an extra step for accumulate but we don't support
        # it yet
        add_reduceat = self._reduceat(np.add)

        for with_cast in (True, False):
            arr = np.arange(5)
            out = np.arange(3)  # definitely wrong shape
            if with_cast:
                # If a cast is necessary on the output, we can be sure to use
                # the generic NpyIter (non-fast) path.
                out = out.astype(np.float64)

            with self.assertRaises(ValueError):
                add_reduceat(arr, [0, 3], out=out)

    def test_reduceat_empty(self):
        """Reduceat should work with empty arrays"""
        indices = np.array([], 'i4')
        x = np.array([], 'f8')
        add_reduceat = self._reduceat(np.add)
        expected = np.add.reduceat(x, indices)
        got = add_reduceat(x, indices)
        self.assertPreciseEqual(expected, got)
        self.assertEqual(expected.dtype, got.dtype)

        # Another case with a slightly different zero-sized shape
        x = np.ones((5, 2))
        idx = np.asarray([], dtype=np.intp)
        self._compare_output(np.add, x, idx, axis=0)
        self._compare_output(np.add, x, idx, axis=1)

    def test_reduceat_error_ndim_2(self):
        add_reduceat = self._reduceat(np.add)

        # indices ndim > 1
        a = np.arange(5)
        idx = np.arange(10).reshape(5, 2)
        with self.assertRaisesRegex(TypingError, 'have at most 1 dimension'):
            add_reduceat(a, idx)

    def test_reduceat_error_non_binary_function(self):
        # non binary functions
        @vectorize
        def neg(a):
            return np.negative(a)

        @njit
        def neg_reduceat(a, idx):
            return neg.reduceat(a, idx)

        a = np.arange(5)
        msg = 'reduceat only supported for binary functions'
        with self.assertRaisesRegex(TypingError, msg):
            neg_reduceat(a, [0, 1, 2])

    def test_reduceat_error_argument_types(self):
        # first argument must be array-like
        add_reduceat = self._reduceat(np.add)
        with self.assertRaisesRegex(TypingError, '"array" must be array-like'):
            add_reduceat('abc', [1, 2, 3])

        with self.assertRaisesRegex(TypingError, '"indices" must be array-like'):  # noqa: E501
            add_reduceat(np.arange(5), 'abcd')

        with self.assertRaisesRegex(TypingError, 'output must be an array'):
            add_reduceat(np.arange(5), [1, 2, 3], out=())

        with self.assertRaisesRegex(TypingError, '"axis" must be an integer'):
            add_reduceat(np.arange(5), [1, 2, 3], axis=(1,))


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
