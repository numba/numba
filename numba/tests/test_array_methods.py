from __future__ import division

from itertools import product
import sys

import numpy as np

from numba import unittest_support as unittest
from numba import typeof, types
from numba.compiler import compile_isolated
from numba.numpy_support import as_dtype, strict_ufunc_typing
from .support import TestCase, CompilationCache, MemoryLeak, MemoryLeakMixin


def np_around_array(arr, decimals, out):
    np.around(arr, decimals, out)

def np_around_binary(val, decimals):
    return np.around(val, decimals)

def np_around_unary(val):
    return np.around(val)

def np_round_array(arr, decimals, out):
    np.round(arr, decimals, out)

def np_round_binary(val, decimals):
    return np.round(val, decimals)

def np_round_unary(val):
    return np.round(val)

def _fixed_np_round(arr, decimals=0, out=None):
    """
    A slightly bugfixed version of np.round().
    """
    if out is not None and arr.dtype.kind == 'c':
        # workaround for https://github.com/numpy/numpy/issues/5779
        _fixed_np_round(arr.real, decimals, out.real)
        _fixed_np_round(arr.imag, decimals, out.imag)
        return out
    else:
        res = np.round(arr, decimals, out)
        if out is None:
            # workaround for https://github.com/numpy/numpy/issues/5780
            def fixup_signed_zero(arg, res):
                if res == 0.0 and arg < 0:
                    return -np.abs(res)
                else:
                    return res
            if isinstance(arr, (complex, np.complexfloating)):
                res = complex(fixup_signed_zero(arr.real, res.real),
                              fixup_signed_zero(arr.imag, res.imag))
            else:
                res = fixup_signed_zero(arr, res)
        return res


def array_T(arr):
    return arr.T

def array_transpose(arr):
    return arr.transpose()

def array_copy(arr):
    return arr.copy()

def array_reshape(arr, newshape):
    return arr.reshape(newshape)

def array_view(arr, newtype):
    return arr.view(newtype)

# XXX Can't pass a dtype as a Dispatcher argument for now
def make_array_view(newtype):
    def array_view(arr):
        return arr.view(newtype)
    return array_view

def array_sliced_view(arr, ):
    return arr[0:4].view(np.float32)[0]

def np_frombuffer(b):
    """
    np.frombuffer() on a Python-allocated buffer.
    """
    return np.frombuffer(b)

def np_frombuffer_dtype(b):
    return np.frombuffer(b, dtype=np.complex64)

def np_frombuffer_allocated(shape):
    """
    np.frombuffer() on a Numba-allocated buffer.
    """
    arr = np.ones(shape, dtype=np.int32)
    return np.frombuffer(arr)

def np_frombuffer_allocated_dtype(shape):
    arr = np.ones(shape, dtype=np.int32)
    return np.frombuffer(arr, dtype=np.complex64)

def identity_usecase(a, b):
    return (a is b), (a is not b)

def array_nonzero(a):
    return a.nonzero()

def np_nonzero(a):
    return np.nonzero(a)

def np_where_1(c):
    return np.where(c)

def np_where_3(c, x, y):
    return np.where(c, x, y)


class TestArrayMethodsCustom(MemoryLeak, TestCase):
    """
    Test np.round, np.around, ndarray.reshape
    """

    def setUp(self):
        super(TestArrayMethodsCustom, self).setUp()
        self.ccache = CompilationCache()

    def check_round_array(self, pyfunc):
        def check_round(cfunc, values, inty, outty, decimals):
            # Create input and output arrays of the right type
            arr = values.astype(as_dtype(inty))
            out = np.zeros_like(arr).astype(as_dtype(outty))
            pyout = out.copy()
            _fixed_np_round(arr, decimals, pyout)
            self.memory_leak_setup()
            cfunc(arr, decimals, out)
            self.memory_leak_teardown()
            np.testing.assert_allclose(out, pyout)
            # Output shape mismatch
            with self.assertRaises(ValueError) as raises:
                cfunc(arr, decimals, out[1:])
            self.assertEqual(str(raises.exception),
                             "invalid output shape")

        def check_types(argtypes, outtypes, values):
            for inty, outty in product(argtypes, outtypes):
                cres = compile_isolated(pyfunc,
                                        (types.Array(inty, 1, 'A'),
                                         types.int32,
                                         types.Array(outty, 1, 'A')))
                cfunc = cres.entry_point
                check_round(cres.entry_point, values, inty, outty, 0)
                check_round(cres.entry_point, values, inty, outty, 1)
                if not isinstance(outty, types.Integer):
                    check_round(cres.entry_point, values * 10, inty, outty, -1)
                else:
                    # Avoid Numpy bug when output is an int:
                    # https://github.com/numpy/numpy/issues/5777
                    pass

        values = np.array([-3.0, -2.5, -2.25, -1.5, 1.5, 2.25, 2.5, 2.75])

        if strict_ufunc_typing:
            argtypes = (types.float64, types.float32)
        else:
            argtypes = (types.float64, types.float32, types.int32)
        check_types(argtypes, argtypes, values)

        argtypes = (types.complex64, types.complex128)
        check_types(argtypes, argtypes, values * (1 - 1j))

    def test_round_array(self):
        self.check_round_array(np_round_array)

    def test_around_array(self):
        self.check_round_array(np_around_array)

    def test_array_reshape(self):
        pyfunc = array_reshape
        def run(arr, shape):
            cres = self.ccache.compile(pyfunc, (typeof(arr), typeof(shape)))
            return cres.entry_point(arr, shape)
        def check(arr, shape):
            expected = pyfunc(arr, shape)
            self.memory_leak_setup()
            got = run(arr, shape)
            self.assertPreciseEqual(got, expected)
            del got
            self.memory_leak_teardown()
        def check_err_shape(arr, shape):
            with self.assertRaises(NotImplementedError) as raises:
                run(arr, shape)
            self.assertEqual(str(raises.exception),
                             "incompatible shape for array")
        def check_err_size(arr, shape):
            with self.assertRaises(ValueError) as raises:
                run(arr, shape)
            self.assertEqual(str(raises.exception),
                             "total size of new array must be unchanged")

        def check_err_negative(arr, shape):
            with self.assertRaises(NotImplementedError) as raises:
                run(arr, shape)
            self.assertEqual(str(raises.exception),
                             "negative shape is not handled, yet")

        # C-contiguous
        arr = np.arange(24)
        check(arr, (24,))
        check(arr, (4, 6))
        check(arr, (8, 3))
        check(arr, (8, 1, 3))
        check(arr, (1, 8, 1, 1, 3, 1))
        arr = np.arange(24).reshape((2, 3, 4))
        check(arr, (24,))
        check(arr, (4, 6))
        check(arr, (8, 3))
        check(arr, (8, 1, 3))
        check(arr, (1, 8, 1, 1, 3, 1))
        check_err_size(arr, (25,))
        check_err_size(arr, (8, 4))
        arr = np.arange(24).reshape((1, 8, 1, 1, 3, 1))
        check(arr, (24,))
        check(arr, (4, 6))
        check(arr, (8, 3))
        check(arr, (8, 1, 3))

        # F-contiguous
        arr = np.arange(24).reshape((2, 3, 4)).T
        check(arr, (4, 3, 2))
        check(arr, (1, 4, 1, 3, 1, 2, 1))
        check_err_shape(arr, (2, 3, 4))
        check_err_shape(arr, (6, 4))
        check_err_shape(arr, (2, 12))

        # Test unhandled value (negative shape)
        arr = np.arange(25).reshape(5,5)
        check_err_negative(arr, -1)
        check_err_negative(arr, (-1,))
        check_err_negative(arr, (-1, -2, 5, 5))

    def test_array_view(self):

        def run(arr, dtype):
            pyfunc = make_array_view(dtype)
            cres = self.ccache.compile(pyfunc, (typeof(arr),))
            return cres.entry_point(arr)
        def check(arr, dtype):
            expected = arr.view(dtype)
            self.memory_leak_setup()
            got = run(arr, dtype)
            self.assertPreciseEqual(got, expected)
            del got
            self.memory_leak_teardown()
        def check_err(arr, dtype):
            with self.assertRaises(ValueError) as raises:
                run(arr, dtype)
            self.assertEqual(str(raises.exception),
                             "new type not compatible with array")

        dt1 = np.dtype([('a', np.int8), ('b', np.int8)])
        dt2 = np.dtype([('u', np.int16), ('v', np.int8)])
        dt3 = np.dtype([('x', np.int16), ('y', np.int16)])

        # C-contiguous
        arr = np.arange(24, dtype=np.int8)
        check(arr, np.dtype('int16'))
        check(arr, np.int16)
        check(arr, np.int8)
        check(arr, np.float32)
        check(arr, np.complex64)
        check(arr, dt1)
        check(arr, dt2)
        check_err(arr, np.complex128)

        # Last dimension must have a compatible size
        arr = arr.reshape((3, 8))
        check(arr, np.int8)
        check(arr, np.float32)
        check(arr, np.complex64)
        check(arr, dt1)
        check_err(arr, dt2)
        check_err(arr, np.complex128)

        # F-contiguous
        arr = np.arange(24, dtype=np.int8).reshape((3, 8)).T
        check(arr, np.int8)
        check(arr, np.float32)
        check(arr, np.complex64)
        check(arr, dt1)
        check_err(arr, dt2)
        check_err(arr, np.complex128)

        # Non-contiguous: only a type with the same itemsize can be used
        arr = np.arange(16, dtype=np.int32)[::2]
        check(arr, np.uint32)
        check(arr, np.float32)
        check(arr, dt3)
        check_err(arr, np.int8)
        check_err(arr, np.int16)
        check_err(arr, np.int64)
        check_err(arr, dt1)
        check_err(arr, dt2)

        # Zero-dim array: only a type with the same itemsize can be used
        arr = np.array([42], dtype=np.int32).reshape(())
        check(arr, np.uint32)
        check(arr, np.float32)
        check(arr, dt3)
        check_err(arr, np.int8)
        check_err(arr, np.int16)
        check_err(arr, np.int64)
        check_err(arr, dt1)
        check_err(arr, dt2)

    def test_array_sliced_view(self):
        """
        Test .view() on A layout array but has contiguous innermost dimension.
        """
        pyfunc = array_sliced_view
        cres = self.ccache.compile(pyfunc, (types.uint8[:],))
        cfunc = cres.entry_point

        orig = np.array([1.5, 2], dtype=np.float32)
        byteary = orig.view(np.uint8)

        expect = pyfunc(byteary)
        got = cfunc(byteary)

        self.assertEqual(expect, got)

    @unittest.skipIf(sys.version_info < (2, 7),
                     "buffer protocol not supported on Python 2.6")
    def check_np_frombuffer(self, pyfunc):
        def run(buf):
            cres = self.ccache.compile(pyfunc, (typeof(buf),))
            return cres.entry_point(buf)
        def check(buf):
            old_refcnt = sys.getrefcount(buf)
            expected = pyfunc(buf)
            self.memory_leak_setup()
            got = run(buf)
            self.assertPreciseEqual(got, expected)
            del expected
            self.assertEqual(sys.getrefcount(buf), old_refcnt + 1)
            del got
            self.assertEqual(sys.getrefcount(buf), old_refcnt)
            self.memory_leak_teardown()

        b = bytearray(range(16))
        check(b)
        if sys.version_info >= (3,):
            check(bytes(b))
            check(memoryview(b))
        check(np.arange(12))
        b = np.arange(12).reshape((3, 4))
        check(b)

        with self.assertRaises(ValueError) as raises:
            run(bytearray(b"xxx"))
        self.assertEqual("buffer size must be a multiple of element size",
                         str(raises.exception))

    def test_np_frombuffer(self):
        self.check_np_frombuffer(np_frombuffer)

    def test_np_frombuffer_dtype(self):
        self.check_np_frombuffer(np_frombuffer_dtype)


class TestArrayMethods(MemoryLeakMixin, TestCase):
    """
    Test various array methods and array-related functions.
    """

    def setUp(self):
        super(TestArrayMethods, self).setUp()
        self.ccache = CompilationCache()

    def check_round_scalar(self, unary_pyfunc, binary_pyfunc):
        base_values = [-3.0, -2.5, -2.25, -1.5, 1.5, 2.25, 2.5, 2.75]
        complex_values = [x * (1 - 1j) for x in base_values]
        int_values = [int(x) for x in base_values]
        argtypes = (types.float64, types.float32, types.int32,
                    types.complex64, types.complex128)
        argvalues = [base_values, base_values, int_values,
                     complex_values, complex_values]

        pyfunc = binary_pyfunc
        for ty, values in zip(argtypes, argvalues):
            cres = compile_isolated(pyfunc, (ty, types.int32))
            cfunc = cres.entry_point
            for decimals in (1, 0, -1):
                for v in values:
                    if decimals > 0:
                        v *= 10
                    expected = _fixed_np_round(v, decimals)
                    got = cfunc(v, decimals)
                    self.assertPreciseEqual(got, expected)

        pyfunc = unary_pyfunc
        for ty, values in zip(argtypes, argvalues):
            cres = compile_isolated(pyfunc, (ty,))
            cfunc = cres.entry_point
            for v in values:
                expected = _fixed_np_round(v)
                got = cfunc(v)
                self.assertPreciseEqual(got, expected)

    def test_round_scalar(self):
        self.check_round_scalar(np_round_unary, np_round_binary)

    def test_around_scalar(self):
        self.check_round_scalar(np_around_unary, np_around_binary)

    def check_layout_dependent_func(self, pyfunc, fac=np.arange):
        def check_arr(arr):
            cres = compile_isolated(pyfunc, (typeof(arr),))
            self.assertPreciseEqual(cres.entry_point(arr), pyfunc(arr))
        arr = fac(24)
        check_arr(arr)
        check_arr(arr.reshape((3, 8)))
        check_arr(arr.reshape((3, 8)).T)
        check_arr(arr.reshape((3, 8))[::2])
        check_arr(arr.reshape((2, 3, 4)))
        check_arr(arr.reshape((2, 3, 4)).T)
        check_arr(arr.reshape((2, 3, 4))[::2])
        arr = np.array([0]).reshape(())
        check_arr(arr)

    def test_array_transpose(self):
        self.check_layout_dependent_func(array_transpose)

    def test_array_T(self):
        self.check_layout_dependent_func(array_T)

    def test_array_copy(self):
        self.check_layout_dependent_func(array_copy)

    @unittest.skipIf(sys.version_info < (2, 7),
                     "buffer protocol not supported on Python 2.6")
    def check_np_frombuffer_allocated(self, pyfunc):
        def run(shape):
            cres = self.ccache.compile(pyfunc, (typeof(shape),))
            return cres.entry_point(shape)
        def check(shape):
            expected = pyfunc(shape)
            got = run(shape)
            self.assertPreciseEqual(got, expected)

        check((16,))
        check((4, 4))
        check((1, 0, 1))

    def test_np_frombuffer_allocated(self):
        self.check_np_frombuffer_allocated(np_frombuffer_allocated)

    def test_np_frombuffer_allocated(self):
        self.check_np_frombuffer_allocated(np_frombuffer_allocated_dtype)

    def check_nonzero(self, pyfunc):
        def fac(N):
            np.random.seed(42)
            arr = np.random.random(N)
            arr[arr < 0.3] = 0.0
            arr[arr > 0.7] = float('nan')
            return arr

        def check_arr(arr):
            cres = compile_isolated(pyfunc, (typeof(arr),))
            expected = pyfunc(arr)
            # NOTE: Numpy 1.9 returns readonly arrays for multidimensional
            # arrays.  Workaround this by copying the results.
            expected = [a.copy() for a in expected]
            self.assertPreciseEqual(cres.entry_point(arr), expected)

        arr = np.int16([1, 0, -1, 0])
        check_arr(arr)
        arr = np.bool_([1, 0, 1])
        check_arr(arr)

        arr = fac(24)
        check_arr(arr)
        check_arr(arr.reshape((3, 8)))
        check_arr(arr.reshape((3, 8)).T)
        check_arr(arr.reshape((3, 8))[::2])
        check_arr(arr.reshape((2, 3, 4)))
        check_arr(arr.reshape((2, 3, 4)).T)
        check_arr(arr.reshape((2, 3, 4))[::2])
        for v in (0.0, 1.5, float('nan')):
            arr = np.array([v]).reshape(())
            check_arr(arr)

    def test_array_nonzero(self):
        self.check_nonzero(array_nonzero)

    def test_np_nonzero(self):
        self.check_nonzero(np_nonzero)

    def test_np_where_1(self):
        self.check_nonzero(np_where_1)

    def test_np_where_3(self):
        pyfunc = np_where_3
        def fac(N):
            np.random.seed(42)
            arr = np.random.random(N)
            arr[arr < 0.3] = 0.0
            arr[arr > 0.7] = float('nan')
            return arr

        def check_arr(arr):
            x = np.zeros_like(arr, dtype=np.float64)
            y = np.copy(x)
            x.fill(4)
            y.fill(9)
            cres = compile_isolated(pyfunc, (typeof(arr), typeof(x), typeof(y)))
            expected = pyfunc(arr, x, y)
            got = cres.entry_point(arr, x, y)
            # Contiguity of result varies accross Numpy versions, only
            # check contents.
            self.assertEqual(got.dtype, expected.dtype)
            np.testing.assert_array_equal(got, expected)

        def check_scal(scal):
            x = 4
            y = 5
            cres = compile_isolated(pyfunc, (typeof(scal), typeof(x), typeof(y)))
            expected = pyfunc(scal, x, y)
            got = cres.entry_point(scal, x, y)
            self.assertPreciseEqual(got, expected)

        arr = np.int16([1, 0, -1, 0])
        check_arr(arr)
        arr = np.bool_([1, 0, 1])
        check_arr(arr)

        arr = fac(24)
        check_arr(arr)
        check_arr(arr.reshape((3, 8)))
        check_arr(arr.reshape((3, 8)).T)
        check_arr(arr.reshape((3, 8))[::2])
        check_arr(arr.reshape((2, 3, 4)))
        check_arr(arr.reshape((2, 3, 4)).T)
        check_arr(arr.reshape((2, 3, 4))[::2])
        for v in (0.0, 1.5, float('nan')):
            arr = np.array([v]).reshape(())
            check_arr(arr)

        for x in (0, 1, True, False, 2.5, 0j):
            check_scal(x)


class TestArrayComparisons(TestCase):

    def test_identity(self):
        def check(a, b, expected):
            cres = compile_isolated(pyfunc, (typeof(a), typeof(b)))
            self.assertPreciseEqual(cres.entry_point(a, b),
                                    (expected, not expected))

        pyfunc = identity_usecase

        arr = np.zeros(10, dtype=np.int32).reshape((2, 5))
        check(arr, arr, True)
        check(arr, arr[:], True)
        check(arr, arr.copy(), False)
        check(arr, arr.view('uint32'), False)
        check(arr, arr.T, False)
        check(arr, arr[:-1], False)

    # Other comparison operators ('==', etc.) are tested in test_ufuncs


if __name__ == '__main__':
    unittest.main()
