from __future__ import print_function
import numpy as np

from numba.compiler import compile_isolated, Flags
from numba import types, from_dtype, errors, typeof
import numba.unittest_support as unittest
from numba.tests.support import TestCase, MemoryLeakMixin

enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

no_pyobj_flags = Flags()
no_pyobj_flags.set('nrt')


def reshape_array(a):
    return a.reshape(3, 3)


def reshape_array_to_1d(a):
    return a.reshape(a.size)


def flatten_array(a):
    return a.flatten()


def ravel_array(a):
    return a.ravel()


def ravel_array_size(a):
    return a.ravel().size


def transpose_array(a):
    return a.transpose()


def squeeze_array(a):
    return a.squeeze()


def convert_array_str(a):
    # astype takes no kws argument in numpy1.6
    return a.astype('f4')


def convert_array_dtype(a):
    # astype takes no kws argument in numpy1.6
    return a.astype(np.float32)


def add_axis1(a):
    return np.expand_dims(a, axis=0)


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
    def test_reshape_array(self, flags=enable_pyobj_flags):
        a = np.arange(9)

        pyfunc = reshape_array
        arraytype1 = typeof(a)
        cr = compile_isolated(pyfunc, (arraytype1,), flags=flags)
        cfunc = cr.entry_point

        expected = pyfunc(a)
        got = cfunc(a)
        np.testing.assert_equal(expected, got)

    def test_reshape_array_npm(self):
        self.test_reshape_array(flags=no_pyobj_flags)

    def test_reshape_array_to_1d(self, flags=enable_pyobj_flags,
                                 layout='C'):
        a = np.arange(9).reshape(3, 3)
        if layout == 'F':
            a = a.T

        pyfunc = reshape_array_to_1d
        arraytype1 = typeof(a)
        if layout == 'A':
            # Force A layout
            arraytype1 = arraytype1.copy(layout='A')

        self.assertEqual(arraytype1.layout, layout)
        cr = compile_isolated(pyfunc, (arraytype1,), flags=flags)
        cfunc = cr.entry_point

        expected = pyfunc(a)
        got = cfunc(a)
        self.assertEqual(got.ndim, 1)
        np.testing.assert_equal(expected, got)

    def test_reshape_array_to_1d_npm(self):
        self.test_reshape_array_to_1d(flags=no_pyobj_flags)
        with self.assertRaises(NotImplementedError) as raises:
            self.test_reshape_array_to_1d(flags=no_pyobj_flags, layout='F')
        self.assertIn("incompatible shape for array", str(raises.exception))
        with self.assertTypingError() as raises:
            # The following will leak due to lack of post exception cleanup
            self.test_reshape_array_to_1d(flags=no_pyobj_flags, layout='A')
        self.assertIn("reshape() supports contiguous array only",
                      str(raises.exception))
        # Disable leak check for the last `test_reshape_array_to_1d` call.
        self.disable_leak_check()

    @unittest.expectedFailure
    def test_reshape_array_to_1d_leak_error_npm(self):
        """
        Rerun the test in ``test_reshape_array_to_1d_npm`` that will cause
        a leak error.
        """
        with self.assertRaises(NotImplementedError) as raises:
            self.test_reshape_array_to_1d(flags=no_pyobj_flags, layout='F')
        self.assertIn("incompatible shape for array", str(raises.exception))
        # The leak check is not captured by the expectedFailure.
        # We need to disable it because `test_reshape_array_to_1d` will leak
        # due to the lack of post exception cleanup
        self.disable_leak_check()
        # The following checks for memory leak and it will fail.
        # This will trigger the expectedFailure
        self.assert_no_memory_leak()

    def test_flatten_array(self, flags=enable_pyobj_flags):
        a = np.arange(9).reshape(3, 3)

        pyfunc = flatten_array
        arraytype1 = typeof(a)
        cr = compile_isolated(pyfunc, (arraytype1,), flags=flags)
        cfunc = cr.entry_point

        expected = pyfunc(a)
        got = cfunc(a)
        np.testing.assert_equal(expected, got)

    def test_flatten_array_npm(self):
        with self.assertRaises(errors.UntypedAttributeError) as raises:
            self.test_flatten_array(flags=no_pyobj_flags)

        self.assertIn("flatten", str(raises.exception))

    def test_ravel_array(self, flags=enable_pyobj_flags):
        a = np.arange(9).reshape(3, 3)

        pyfunc = ravel_array
        arraytype1 = typeof(a)
        cr = compile_isolated(pyfunc, (arraytype1,), flags=flags)
        cfunc = cr.entry_point

        expected = pyfunc(a)
        got = cfunc(a)
        np.testing.assert_equal(expected, got)

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
        with self.assertRaises(errors.UntypedAttributeError) as raises:
            self.test_ravel_array(flags=no_pyobj_flags)

        self.assertIn("ravel", str(raises.exception))

    def test_ravel_array_size_npm(self):
        with self.assertRaises(errors.UntypedAttributeError) as raises:
            self.test_ravel_array_size(flags=no_pyobj_flags)

        self.assertIn("ravel", str(raises.exception))

    def test_transpose_array(self, flags=enable_pyobj_flags):
        a = np.arange(9).reshape(3, 3)

        pyfunc = transpose_array
        arraytype1 = typeof(a)
        cr = compile_isolated(pyfunc, (arraytype1,), flags=flags)
        cfunc = cr.entry_point

        expected = pyfunc(a)
        got = cfunc(a)
        np.testing.assert_equal(expected, got)

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
        with self.assertRaises(errors.UntypedAttributeError) as raises:
            self.test_squeeze_array(flags=no_pyobj_flags)

        self.assertIn("squeeze", str(raises.exception))

    def test_convert_array_str(self, flags=enable_pyobj_flags):
        a = np.arange(9, dtype='i4')

        pyfunc = convert_array_str
        arraytype1 = typeof(a)
        cr = compile_isolated(pyfunc, (arraytype1,), flags=flags)
        cfunc = cr.entry_point

        expected = pyfunc(a)
        got = cfunc(a)
        self.assertPreciseEqual(expected, got)

    def test_convert_array_str_npm(self):
        with self.assertRaises(errors.UntypedAttributeError) as raises:
            self.test_convert_array_str(flags=no_pyobj_flags)

        self.assertIn("astype", str(raises.exception))

    def test_convert_array(self, flags=enable_pyobj_flags):
        a = np.arange(9, dtype='i4')

        pyfunc = convert_array_dtype
        arraytype1 = typeof(a)
        cr = compile_isolated(pyfunc, (arraytype1,), flags=flags)
        cfunc = cr.entry_point

        expected = pyfunc(a)
        got = cfunc(a)
        np.testing.assert_equal(expected, got)

    def test_convert_array_npm(self):
        with self.assertRaises(errors.UntypedAttributeError) as raises:
            self.test_convert_array(flags=no_pyobj_flags)

        self.assertIn("astype", str(raises.exception))

    def test_add_axis1(self, flags=enable_pyobj_flags):
        a = np.arange(9).reshape(3, 3)

        pyfunc = add_axis1
        arraytype1 = typeof(a)
        cr = compile_isolated(pyfunc, (arraytype1,), flags=flags)
        cfunc = cr.entry_point

        expected = pyfunc(a)
        got = cfunc(a)
        np.testing.assert_equal(expected, got)

    def test_add_axis1_npm(self):
        with self.assertRaises(errors.UntypedAttributeError) as raises:
            self.test_add_axis1(flags=no_pyobj_flags)

        self.assertIn("expand_dims", str(raises.exception))

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
