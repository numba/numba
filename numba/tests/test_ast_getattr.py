#! /usr/bin/env python
# ______________________________________________________________________

from numba.translate import _plat_bits
from numba.decorators import autojit

import numpy as np
import numpy

import unittest

# ______________________________________________________________________

@autojit(backend='ast')
def get_ndarray_ndim(ndarr):
    return ndarr.ndim

@autojit(backend='ast')
def get_ndarray_shape(ndarr):
    return ndarr.shape

@autojit(backend='ast')
def get_ndarray_data(ndarr):
    return ndarr.data

@autojit(backend='ast')
def get_ndarray_2_shape_unpack_0(ndarr):
    dim0, _ = ndarr.shape
    return dim0

@autojit(backend='ast')
def get_ndarray_2_shape_unpack_1(ndarr):
    _, dim1 = ndarr.shape
    return dim1

# ______________________________________________________________________

class TestGetattr(unittest.TestCase):
    def test_getattr_ndim(self):
        result = get_ndarray_ndim(np.empty((2,)))
        self.assertEqual(result, 1)
        result = get_ndarray_ndim(np.empty((2, 2)))
        self.assertEqual(result, 2)

    def test_getattr_shape(self):
        # This is broken since the shape is a ctypes array, and the shape array
        # doesn't hold a reference to the ndarray!
        result = get_ndarray_shape(np.empty((10,)))
        self.assertEqual(result[0], 10)

        result = get_ndarray_shape(np.empty((10, 20)))
        self.assertEqual(result[0], 10)
        self.assertEqual(result[1], 10)

    def test_getattr_shape_unpack(self):
        array = np.empty((1, 2))
        dim0 = get_ndarray_2_shape_unpack_0(array)
        dim1 = get_ndarray_2_shape_unpack_1(array)
        self.assertEqual((dim0, dim1), (1, 2))

    def test_getattr_data_1(self):
        test_data = numpy.array([1., 2., 3.])
        data_pointer = get_ndarray_data(test_data)
        self.assertEqual(data_pointer[0], 1.)
        self.assertEqual(data_pointer[1], 2.)
        self.assertEqual(data_pointer[2], 3.)

    def test_getattr_data_2(self):
        test_data = numpy.array([[1., 2., 3.], [4., 5., 6.]])
        result = get_ndarray_data(test_data)
        self.assertEqual(result[0], 1.)
        self.assertEqual(result[1], 2.)
        self.assertEqual(result[2], 3.)
        self.assertEqual(result[3], 4.)
        self.assertEqual(result[4], 5.)
        self.assertEqual(result[5], 6.)

# ______________________________________________________________________

if __name__ == "__main__":
    TestGetattr('test_getattr_data_1').debug()
#    unittest.main()

# ______________________________________________________________________
# End of test_ast_getattr.py
