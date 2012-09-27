#! /usr/bin/env python
# ______________________________________________________________________

from numba.translate import _plat_bits
from numba.decorators import jit

import numpy

import unittest

# ______________________________________________________________________

def get_ndarray_ndim(ndarr):
    return ndarr.ndim

def get_ndarray_shape(ndarr):
    return ndarr.shape

def get_ndarray_data(ndarr):
    return ndarr.data

def get_ndarray_2_shape_unpack_0(ndarr):
    dim0, _ = ndarr.shape
    return dim0

def get_ndarray_2_shape_unpack_1(ndarr):
    _, dim1 = ndarr.shape
    return dim1

# ______________________________________________________________________

class TestGetattr(unittest.TestCase):
    def test_getattr_ndim_1(self):
        test_data1 = numpy.array([1., 2., 3.])
        compiled_fn1 = jit(restype = 'i',
                                    argtypes = [['d']])(get_ndarray_ndim)
        self.assertEqual(compiled_fn1(test_data1), 1)

    def test_getattr_ndim_2(self):
        test_data2 = numpy.array([[1., 2., 3.], [4., 5., 6.]])
        compiled_fn2 = jit(restype = 'i',
                                     argtypes = [[['d']]])(get_ndarray_ndim)
        self.assertEqual(compiled_fn2(test_data2), 2)

    def test_getattr_shape_1(self):
        test_data = numpy.array([1., 2., 3.])
        compiled_fn = jit(restype = 'i%d*' % (_plat_bits // 8),
                                    argtypes = [['d']])(get_ndarray_shape)
        result = compiled_fn(test_data)
        self.assertEqual(result[0], 3)

    def test_getattr_shape_2(self):
        test_data2 = numpy.array([[1., 2., 3.], [4., 5., 6.]])
        compiled_fn2 = jit(restype = 'i%d*' % (_plat_bits // 8),
                                     argtypes = [[['d']]])(get_ndarray_shape)
        result = compiled_fn2(test_data2)
        self.assertEqual(result[0], 2)
        self.assertEqual(result[1], 3)

    def test_getattr_shape_2_unpack(self):
        compiler_fn = jit(restype = 'i%d' % (_plat_bits // 8),
                                    argtypes = [[['d']]])
        dim0_fn, dim1_fn = (compiler_fn(fn) 
                            for fn in (get_ndarray_2_shape_unpack_0,
                                       get_ndarray_2_shape_unpack_1))
        test_data2 = numpy.array([[1., 2., 3.], [4., 5., 6.]])
        self.assertEqual(dim0_fn(test_data2), 2)
        self.assertEqual(dim1_fn(test_data2), 3)

    def test_getattr_data_1(self):
        test_data = numpy.array([1., 2., 3.])
        compiled_fn = jit(restype = 'd*',
                                    argtypes = [['d']])(get_ndarray_data)
        result = compiled_fn(test_data)
        self.assertEqual(result[0], 1.)
        self.assertEqual(result[1], 2.)
        self.assertEqual(result[2], 3.)

    def test_getattr_data_2(self):
        test_data = numpy.array([[1., 2., 3.], [4., 5., 6.]])
        compiled_fn = jit(restype = 'd*',
                                    argtypes = [[['d']]])(get_ndarray_data)
        result = compiled_fn(test_data)
        self.assertEqual(result[0], 1.)
        self.assertEqual(result[1], 2.)
        self.assertEqual(result[2], 3.)
        self.assertEqual(result[3], 4.)
        self.assertEqual(result[4], 5.)
        self.assertEqual(result[5], 6.)

# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main()

# ______________________________________________________________________
# End of test_getattr.py
