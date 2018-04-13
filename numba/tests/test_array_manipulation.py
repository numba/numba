from __future__ import print_function

from functools import partial
from itertools import permutations
import numba.unittest_support as unittest

import numpy as np

from numba.compiler import compile_isolated, Flags
from numba import jit, types, from_dtype, errors, typeof
from numba.errors import TypingError
from .support import TestCase, MemoryLeakMixin, CompilationCache, tag

enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

no_pyobj_flags = Flags()
no_pyobj_flags.set('nrt')


def from_generic(pyfuncs_to_use):
    """Decorator for generic check functions.
        Iterates over 'pyfuncs_to_use', calling 'func' with the iterated
        item as first argument. Example:

        @from_generic(numpy_array_reshape, array_reshape)
        def check_only_shape(pyfunc, arr, shape, expected_shape):
            # Only check Numba result to avoid Numpy bugs
            self.memory_leak_setup()
            got = generic_run(pyfunc, arr, shape)
            self.assertEqual(got.shape, expected_shape)
            self.assertEqual(got.size, arr.size)
            del got
            self.memory_leak_teardown()
    """
    def decorator(func):
        def result(*args, **kwargs):
            return (func(pyfunc, *args, **kwargs) for pyfunc in pyfuncs_to_use)
        return result
    return decorator


def array_reshape(arr, newshape):
    return arr.reshape(newshape)


def numpy_array_reshape(arr, newshape):
    return np.reshape(arr, newshape)


def flatten_array(a):
    return a.flatten()


def ravel_array(a):
    return a.ravel()


def ravel_array_size(a):
    return a.ravel().size


def numpy_ravel_array(a):
    return np.ravel(a)


def transpose_array(a):
    return a.transpose()


def numpy_transpose_array(a):
    return np.transpose(a)

def numpy_transpose_array_axes_kwarg(arr, axes):
    return np.transpose(arr, axes=axes)

def array_transpose_axes(arr, axes):
    return arr.transpose(axes)

def squeeze_array(a):
    return a.squeeze()


def expand_dims(a, axis):
    return np.expand_dims(a, axis)


def atleast_1d(*args):
    return np.atleast_1d(*args)


def atleast_2d(*args):
    return np.atleast_2d(*args)


def atleast_3d(*args):
    return np.atleast_3d(*args)


def as_strided1(a):
    # as_strided() with implicit shape
    strides = (a.strides[0] // 2,) + a.strides[1:]
    return np.lib.stride_tricks.as_strided(a, strides=strides)


def as_strided2(a):
    # Rolling window example as in https://github.com/numba/numba/issues/1884
    window = 3
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def add_axis2(a):
    return a[np.newaxis, :]


def bad_index(arr, arr2d):
    x = arr.x,
    y = arr.y
    # note that `x` is a tuple, which causes a new axis to be created.
    arr2d[x, y] = 1.0


def bad_float_index(arr):
    # 2D index required for this function because 1D index
    # fails typing
    return arr[1, 2.0]


class TestArrayManipulation(MemoryLeakMixin, TestCase):
    """
    Check shape-changing operations on arrays.
    """

    def setUp(self):
        super(TestArrayManipulation, self).setUp()
        self.ccache = CompilationCache()

    @tag('important')
    def test_array_reshape(self):
        pyfuncs_to_use = [array_reshape, numpy_array_reshape]

        def generic_run(pyfunc, arr, shape):
            cres = compile_isolated(pyfunc, (typeof(arr), typeof(shape)))
            return cres.entry_point(arr, shape)

        @from_generic(pyfuncs_to_use)
        def check(pyfunc, arr, shape):
            expected = pyfunc(arr, shape)
            self.memory_leak_setup()
            got = generic_run(pyfunc, arr, shape)
            self.assertPreciseEqual(got, expected)
            del got
            self.memory_leak_teardown()

        @from_generic(pyfuncs_to_use)
        def check_only_shape(pyfunc, arr, shape, expected_shape):
            # Only check Numba result to avoid Numpy bugs
            self.memory_leak_setup()
            got = generic_run(pyfunc, arr, shape)
            self.assertEqual(got.shape, expected_shape)
            self.assertEqual(got.size, arr.size)
            del got
            self.memory_leak_teardown()

        @from_generic(pyfuncs_to_use)
        def check_err_shape(pyfunc, arr, shape):
            with self.assertRaises(NotImplementedError) as raises:
                generic_run(pyfunc, arr, shape)
            self.assertEqual(str(raises.exception),
                             "incompatible shape for array")

        @from_generic(pyfuncs_to_use)
        def check_err_size(pyfunc, arr, shape):
            with self.assertRaises(ValueError) as raises:
                generic_run(pyfunc, arr, shape)
            self.assertEqual(str(raises.exception),
                             "total size of new array must be unchanged")

        @from_generic(pyfuncs_to_use)
        def check_err_multiple_negative(pyfunc, arr, shape):
            with self.assertRaises(ValueError) as raises:
                generic_run(pyfunc, arr, shape)
            self.assertEqual(str(raises.exception),
                             "multiple negative shape values")


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
        check_err_size(arr, ())
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

        # Test negative shape value
        arr = np.arange(25).reshape(5,5)
        check(arr, -1)
        check(arr, (-1,))
        check(arr, (-1, 5))
        check(arr, (5, -1, 5))
        check(arr, (5, 5, -1))
        check_err_size(arr, (-1, 4))
        check_err_multiple_negative(arr, (-1, -2, 5, 5))
        check_err_multiple_negative(arr, (5, 5, -1, -1))

        # 0-sized arrays
        def check_empty(arr):
            check(arr, 0)
            check(arr, (0,))
            check(arr, (1, 0, 2))
            check(arr, (0, 55, 1, 0, 2))
            # -1 is buggy in Numpy with 0-sized arrays
            check_only_shape(arr, -1, (0,))
            check_only_shape(arr, (-1,), (0,))
            check_only_shape(arr, (0, -1), (0, 0))
            check_only_shape(arr, (4, -1), (4, 0))
            check_only_shape(arr, (-1, 0, 4), (0, 0, 4))
            check_err_size(arr, ())
            check_err_size(arr, 1)
            check_err_size(arr, (1, 2))

        arr = np.array([])
        check_empty(arr)
        check_empty(arr.reshape((3, 2, 0)))

        # Exceptions leak references
        self.disable_leak_check()

    def test_array_transpose_axes(self):
        pyfuncs_to_use = [numpy_transpose_array_axes_kwarg,
                          array_transpose_axes]

        def run(pyfunc, arr, axes):
            cres = self.ccache.compile(pyfunc, (typeof(arr), typeof(axes)))
            return cres.entry_point(arr, axes)

        @from_generic(pyfuncs_to_use)
        def check(pyfunc, arr, axes):
            expected = pyfunc(arr, axes)
            got = run(pyfunc, arr, axes)
            self.assertPreciseEqual(got, expected)
            self.assertEqual(got.flags.f_contiguous,
                             expected.flags.f_contiguous)
            self.assertEqual(got.flags.c_contiguous,
                             expected.flags.c_contiguous)

        @from_generic(pyfuncs_to_use)
        def check_err_axis_repeated(pyfunc, arr, axes):
            with self.assertRaises(ValueError) as raises:
                run(pyfunc, arr, axes)
            self.assertEqual(str(raises.exception),
                             "repeated axis in transpose")

        @from_generic(pyfuncs_to_use)
        def check_err_axis_oob(pyfunc, arr, axes):
            with self.assertRaises(ValueError) as raises:
                run(pyfunc, arr, axes)
            self.assertEqual(str(raises.exception),
                             "axis is out of bounds for array of given dimension")

        @from_generic(pyfuncs_to_use)
        def check_err_invalid_args(pyfunc, arr, axes):
            with self.assertRaises((TypeError, TypingError)):
                run(pyfunc, arr, axes)

        arrs = [np.arange(24),
                np.arange(24).reshape(4, 6),
                np.arange(24).reshape(2, 3, 4),
                np.arange(24).reshape(1, 2, 3, 4),
                np.arange(64).reshape(8, 4, 2)[::3,::2,:]]

        for i in range(len(arrs)):
            # First check `None`, the default, which is to reverse dims
            check(arrs[i], None)
            # Check supplied axis permutations
            for axes in permutations(tuple(range(arrs[i].ndim))):
                ndim = len(axes)
                neg_axes = tuple([x - ndim for x in axes])
                check(arrs[i], axes)
                check(arrs[i], neg_axes)

        # Exceptions leak references
        self.disable_leak_check()

        check_err_invalid_args(arrs[1], "foo")
        check_err_invalid_args(arrs[1], ("foo",))
        check_err_invalid_args(arrs[1], 5.3)
        check_err_invalid_args(arrs[2], (1.2, 5))

        check_err_axis_repeated(arrs[1], (0, 0))
        check_err_axis_repeated(arrs[2], (2, 0, 0))
        check_err_axis_repeated(arrs[3], (3, 2, 1, 1))

        check_err_axis_oob(arrs[0], (1,))
        check_err_axis_oob(arrs[0], (-2,))
        check_err_axis_oob(arrs[1], (0, 2))
        check_err_axis_oob(arrs[1], (-3, 2))
        check_err_axis_oob(arrs[1], (0, -3))
        check_err_axis_oob(arrs[2], (3, 1, 2))
        check_err_axis_oob(arrs[2], (-4, 1, 2))
        check_err_axis_oob(arrs[3], (3, 1, 2, 5))
        check_err_axis_oob(arrs[3], (3, 1, 2, -5))


    @tag('important')
    def test_expand_dims(self):
        pyfunc = expand_dims

        def run(arr, axis):
            cres = self.ccache.compile(pyfunc, (typeof(arr), typeof(axis)))
            return cres.entry_point(arr, axis)

        def check(arr, axis):
            expected = pyfunc(arr, axis)
            self.memory_leak_setup()
            got = run(arr, axis)
            self.assertPreciseEqual(got, expected)
            del got
            self.memory_leak_teardown()

        def check_all_axes(arr):
            for axis in range(-arr.ndim - 1, arr.ndim + 1):
                check(arr, axis)

        # 1d
        arr = np.arange(5)
        check_all_axes(arr)
        # 3d (C, F, A)
        arr = np.arange(24).reshape((2, 3, 4))
        check_all_axes(arr)
        check_all_axes(arr.T)
        check_all_axes(arr[::-1])
        # 0d
        arr = np.array(42)
        check_all_axes(arr)

    def check_atleast_nd(self, pyfunc, cfunc):
        def check_result(got, expected):
            # We would like to check the result has the same contiguity,
            # but we can't rely on the "flags" attribute when there are
            # 1-sized dimensions.
            self.assertStridesEqual(got, expected)
            self.assertPreciseEqual(got.flatten(), expected.flatten())

        def check_single(arg):
            check_result(cfunc(arg), pyfunc(arg))

        def check_tuple(*args):
            expected_tuple = pyfunc(*args)
            got_tuple = cfunc(*args)
            self.assertEqual(len(got_tuple), len(expected_tuple))
            for got, expected in zip(got_tuple, expected_tuple):
                check_result(got, expected)

        # 0d
        a1 = np.array(42)
        a2 = np.array(5j)
        check_single(a1)
        check_tuple(a1, a2)
        # 1d
        b1 = np.arange(5)
        b2 = np.arange(6) + 1j
        b3 = b1[::-1]
        check_single(b1)
        check_tuple(b1, b2, b3)
        # 2d
        c1 = np.arange(6).reshape((2, 3))
        c2 = c1.T
        c3 = c1[::-1]
        check_single(c1)
        check_tuple(c1, c2, c3)
        # 3d
        d1 = np.arange(24).reshape((2, 3, 4))
        d2 = d1.T
        d3 = d1[::-1]
        check_single(d1)
        check_tuple(d1, d2, d3)
        # 4d
        e = np.arange(16).reshape((2, 2, 2, 2))
        check_single(e)
        # mixed dimensions
        check_tuple(a1, b2, c3, d2)

    def test_atleast_1d(self):
        pyfunc = atleast_1d
        cfunc = jit(nopython=True)(pyfunc)
        self.check_atleast_nd(pyfunc, cfunc)

    def test_atleast_2d(self):
        pyfunc = atleast_2d
        cfunc = jit(nopython=True)(pyfunc)
        self.check_atleast_nd(pyfunc, cfunc)

    def test_atleast_3d(self):
        pyfunc = atleast_3d
        cfunc = jit(nopython=True)(pyfunc)
        self.check_atleast_nd(pyfunc, cfunc)

    def check_as_strided(self, pyfunc):
        def run(arr):
            cres = self.ccache.compile(pyfunc, (typeof(arr),))
            return cres.entry_point(arr)
        def check(arr):
            expected = pyfunc(arr)
            got = run(arr)
            self.assertPreciseEqual(got, expected)

        arr = np.arange(24)
        check(arr)
        check(arr.reshape((6, 4)))
        check(arr.reshape((4, 1, 6)))

    def test_as_strided(self):
        self.check_as_strided(as_strided1)
        self.check_as_strided(as_strided2)

    def test_flatten_array(self, flags=enable_pyobj_flags, layout='C'):
        a = np.arange(9).reshape(3, 3)
        if layout == 'F':
            a = a.T

        pyfunc = flatten_array
        arraytype1 = typeof(a)
        if layout == 'A':
            # Force A layout
            arraytype1 = arraytype1.copy(layout='A')

        self.assertEqual(arraytype1.layout, layout)
        cr = compile_isolated(pyfunc, (arraytype1,), flags=flags)
        cfunc = cr.entry_point

        expected = pyfunc(a)
        got = cfunc(a)
        np.testing.assert_equal(expected, got)

    def test_flatten_array_npm(self):
        self.test_flatten_array(flags=no_pyobj_flags)
        self.test_flatten_array(flags=no_pyobj_flags, layout='F')
        self.test_flatten_array(flags=no_pyobj_flags, layout='A')

    def test_ravel_array(self, flags=enable_pyobj_flags):
        def generic_check(pyfunc, a, assume_layout):
            # compile
            arraytype1 = typeof(a)
            self.assertEqual(arraytype1.layout, assume_layout)
            cr = compile_isolated(pyfunc, (arraytype1,), flags=flags)
            cfunc = cr.entry_point

            expected = pyfunc(a)
            got = cfunc(a)
            # Check result matches
            np.testing.assert_equal(expected, got)
            # Check copying behavior
            py_copied = (a.ctypes.data != expected.ctypes.data)
            nb_copied = (a.ctypes.data != got.ctypes.data)
            self.assertEqual(py_copied, assume_layout != 'C')
            self.assertEqual(py_copied, nb_copied)

        check_method = partial(generic_check, ravel_array)
        check_function = partial(generic_check, numpy_ravel_array)

        def check(*args, **kwargs):
            check_method(*args, **kwargs)
            check_function(*args, **kwargs)

        # Check 2D
        check(np.arange(9).reshape(3, 3), assume_layout='C')
        check(np.arange(9).reshape(3, 3, order='F'), assume_layout='F')
        check(np.arange(18).reshape(3, 3, 2)[:, :, 0], assume_layout='A')

        # Check 3D
        check(np.arange(18).reshape(2, 3, 3), assume_layout='C')
        check(np.arange(18).reshape(2, 3, 3, order='F'), assume_layout='F')
        check(np.arange(36).reshape(2, 3, 3, 2)[:, :, :, 0], assume_layout='A')

    def test_ravel_array_size(self, flags=enable_pyobj_flags):
        a = np.arange(9).reshape(3, 3)

        pyfunc = ravel_array_size
        arraytype1 = typeof(a)
        cr = compile_isolated(pyfunc, (arraytype1,), flags=flags)
        cfunc = cr.entry_point

        expected = pyfunc(a)
        got = cfunc(a)
        np.testing.assert_equal(expected, got)

    def test_ravel_array_npm(self):
        self.test_ravel_array(flags=no_pyobj_flags)

    def test_ravel_array_size_npm(self):
        self.test_ravel_array_size(flags=no_pyobj_flags)

    def test_transpose_array(self, flags=enable_pyobj_flags):
        @from_generic([transpose_array, numpy_transpose_array])
        def check(pyfunc):
            a = np.arange(9).reshape(3, 3)

            arraytype1 = typeof(a)
            cr = compile_isolated(pyfunc, (arraytype1,), flags=flags)
            cfunc = cr.entry_point

            expected = pyfunc(a)
            got = cfunc(a)
            np.testing.assert_equal(expected, got)

        check()

    def test_transpose_array_npm(self):
        self.test_transpose_array(flags=no_pyobj_flags)

    def test_squeeze_array(self, flags=enable_pyobj_flags):
        a = np.arange(2 * 1 * 3 * 1 * 4).reshape(2, 1, 3, 1, 4)

        pyfunc = squeeze_array
        arraytype1 = typeof(a)
        cr = compile_isolated(pyfunc, (arraytype1,), flags=flags)
        cfunc = cr.entry_point

        expected = pyfunc(a)
        got = cfunc(a)
        np.testing.assert_equal(expected, got)

    def test_squeeze_array_npm(self):
        with self.assertRaises(errors.TypingError) as raises:
            self.test_squeeze_array(flags=no_pyobj_flags)

        self.assertIn("squeeze", str(raises.exception))

    def test_add_axis2(self, flags=enable_pyobj_flags):
        a = np.arange(9).reshape(3, 3)

        pyfunc = add_axis2
        arraytype1 = typeof(a)
        cr = compile_isolated(pyfunc, (arraytype1,), flags=flags)
        cfunc = cr.entry_point

        expected = pyfunc(a)
        got = cfunc(a)
        np.testing.assert_equal(expected, got)

    def test_add_axis2_npm(self):
        with self.assertTypingError() as raises:
            self.test_add_axis2(flags=no_pyobj_flags)
        self.assertIn("unsupported array index type none in",
                      str(raises.exception))

    def test_bad_index_npm(self):
        with self.assertTypingError() as raises:
            arraytype1 = from_dtype(np.dtype([('x', np.int32),
                                              ('y', np.int32)]))
            arraytype2 = types.Array(types.int32, 2, 'C')
            compile_isolated(bad_index, (arraytype1, arraytype2),
                             flags=no_pyobj_flags)
        self.assertIn('unsupported array index type', str(raises.exception))

    def test_bad_float_index_npm(self):
        with self.assertTypingError() as raises:
            compile_isolated(bad_float_index,
                             (types.Array(types.float64, 2, 'C'),))
        self.assertIn('unsupported array index type float64',
                      str(raises.exception))


if __name__ == '__main__':
    unittest.main()
