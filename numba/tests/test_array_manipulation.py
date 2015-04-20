from __future__ import print_function
import numpy as np

from numba.compiler import compile_isolated, Flags
from numba import types, from_dtype, utils
import numba.unittest_support as unittest
from numba.tests import usecases
from numba.tests.support import TestCase

enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

no_pyobj_flags = Flags()


def reshape_array(a, expected):
    return (a.reshape(3, 3) == expected).all()

def flatten_array(a, expected):
    return (a.flatten() == expected).all()

def ravel_array(a, expected):
    return (a.ravel() == expected).all()

def ravel_array_size(a, expected):
    return (a.ravel().size == expected.size)

def transpose_array(a, expected):
    return (a.transpose() == expected).all()

def squeeze_array(a, expected):
    return (a.squeeze() == expected).all()

def convert_array(a, expected):
    # astype takes no kws argument in numpy1.6
    return (a.astype('f4') == expected).all()

def add_axis1(a, expected):
    return np.expand_dims(a, axis=0).shape == expected.shape

def add_axis2(a, expected):
    return a[np.newaxis,:].shape == expected.shape

def bad_index(arr, arr2d):
    x = arr.x,
    y = arr.y
    arr2d[x, y] = 1.0

def bad_float_index(arr):
    # 2D index required for this function because 1D index
    # fails typing
    return arr[1, 2.0]

class TestArrayManipulation(TestCase):

    def test_reshape_array(self, flags=enable_pyobj_flags):
        pyfunc = reshape_array
        arraytype1 = types.Array(types.int32, 1, 'C')
        arraytype2 = types.Array(types.int32, 2, 'C')
        cr = compile_isolated(pyfunc, (arraytype1, arraytype2),
                              flags=flags)
        cfunc = cr.entry_point

        a = np.arange(9)
        expected = np.arange(9).reshape(3, 3)
        self.assertTrue(cfunc(a, expected))

    def test_reshape_array_npm(self):
        with self.assertTypingError():
            self.test_reshape_array(flags=no_pyobj_flags)

    def test_flatten_array(self, flags=enable_pyobj_flags):
        pyfunc = flatten_array
        arraytype1 = types.Array(types.int32, 2, 'C')
        arraytype2 = types.Array(types.int32, 1, 'C')
        cr = compile_isolated(pyfunc, (arraytype1, arraytype2),
                              flags=flags)
        cfunc = cr.entry_point

        a = np.arange(9).reshape(3, 3)
        expected = np.arange(9).reshape(3, 3).flatten()
        self.assertTrue(cfunc(a, expected))

    def test_flatten_array_npm(self):
        with self.assertTypingError():
            self.test_flatten_array(flags=no_pyobj_flags)

    def test_ravel_array(self, flags=enable_pyobj_flags):
        pyfunc = ravel_array
        arraytype1 = types.Array(types.int32, 2, 'C')
        arraytype2 = types.Array(types.int32, 1, 'C')
        cr = compile_isolated(pyfunc, (arraytype1, arraytype2),
                              flags=flags)
        cfunc = cr.entry_point

        a = np.arange(9).reshape(3, 3)
        expected = np.arange(9).reshape(3, 3).ravel()
        self.assertTrue(cfunc(a, expected))

    def test_ravel_array_size(self, flags=enable_pyobj_flags):
        pyfunc = ravel_array_size
        arraytype1 = types.Array(types.int32, 2, 'C')
        arraytype2 = types.Array(types.int32, 1, 'C')
        cr = compile_isolated(pyfunc, (arraytype1, arraytype2),
                              flags=flags)
        cfunc = cr.entry_point

        a = np.arange(9).reshape(3, 3)
        expected = np.arange(9).reshape(3, 3).ravel()
        self.assertTrue(cfunc(a, expected))

    def test_ravel_array_npm(self):
        with self.assertTypingError():
            self.test_ravel_array(flags=no_pyobj_flags)

    def test_transpose_array(self, flags=enable_pyobj_flags):
        pyfunc = transpose_array
        arraytype1 = types.Array(types.int32, 2, 'C')
        arraytype2 = types.Array(types.int32, 2, 'C')
        cr = compile_isolated(pyfunc, (arraytype1, arraytype2),
                              flags=flags)
        cfunc = cr.entry_point

        a = np.arange(9).reshape(3, 3)
        expected = np.arange(9).reshape(3, 3).transpose()
        self.assertTrue(cfunc(a, expected))

    def test_transpose_array_npm(self):
        with self.assertTypingError():
            self.test_transpose_array(flags=no_pyobj_flags)

    def test_squeeze_array(self, flags=enable_pyobj_flags):
        pyfunc = squeeze_array
        arraytype1 = types.Array(types.int32, 2, 'C')
        arraytype2 = types.Array(types.int32, 2, 'C')
        cr = compile_isolated(pyfunc, (arraytype1, arraytype2),
                              flags=flags)
        cfunc = cr.entry_point

        a = np.arange(2*1*3*1*4).reshape(2,1,3,1,4)
        expected = np.arange(2*1*3*1*4).reshape(2,1,3,1,4).squeeze()
        self.assertTrue(cfunc(a, expected))

    def test_squeeze_array_npm(self):
        with self.assertTypingError():
            self.test_squeeze_array(flags=no_pyobj_flags)

    def test_convert_array(self, flags=enable_pyobj_flags):
        pyfunc = convert_array
        arraytype1 = types.Array(types.int32, 1, 'C')
        arraytype2 = types.Array(types.float32, 1, 'C')
        cr = compile_isolated(pyfunc, (arraytype1, arraytype2),
                              flags=flags)
        cfunc = cr.entry_point

        a = np.arange(9, dtype='i4')
        expected = np.arange(9, dtype='f4')
        self.assertTrue(cfunc(a, expected))

    def test_convert_array_npm(self):
        with self.assertTypingError():
            self.test_convert_array(flags=no_pyobj_flags)

    def test_add_axis1(self, flags=enable_pyobj_flags):
        pyfunc = add_axis1
        arraytype1 = types.Array(types.int32, 1, 'C')
        arraytype2 = types.Array(types.float32, 1, 'C')
        cr = compile_isolated(pyfunc, (arraytype1, arraytype2),
                              flags=flags)
        cfunc = cr.entry_point

        a = np.arange(9).reshape(3,3)
        expected = np.arange(9).reshape(1,3,3)
        self.assertTrue(cfunc(a, expected))

    def test_add_axis1_npm(self):
        with self.assertTypingError():
            self.test_add_axis1(flags=no_pyobj_flags)

    def test_add_axis2(self, flags=enable_pyobj_flags):
        pyfunc = add_axis2
        arraytype1 = types.Array(types.int32, 1, 'C')
        arraytype2 = types.Array(types.float32, 1, 'C')
        cr = compile_isolated(pyfunc, (arraytype1, arraytype2),
                              flags=flags)
        cfunc = cr.entry_point

        a = np.arange(9).reshape(3,3)
        expected = np.arange(9).reshape(1,3,3)
        self.assertTrue(cfunc(a, expected))

    def test_add_axis2_npm(self):
        with self.assertTypingError():
            self.test_add_axis2(flags=no_pyobj_flags)

    def test_bad_index_npm(self):
        with self.assertTypingError() as raises:
            arraytype1 = from_dtype(np.dtype([('x', np.int32),
                                              ('y', np.int32)]))
            arraytype2 = types.Array(types.int32, 2, 'C')
            compile_isolated(bad_index, (arraytype1, arraytype2),
                             flags=no_pyobj_flags)
        self.assertIn('is unsupported for indexing', str(raises.exception))

    def test_bad_float_index_npm(self):
        with self.assertTypingError() as raises:
            compile_isolated(bad_float_index,
                             (types.Array(types.float64, 2, 'C'),))
        self.assertIn('Type float', str(raises.exception))
        self.assertIn('is unsupported for indexing', str(raises.exception))


if __name__ == '__main__':
    unittest.main()

