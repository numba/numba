#! /usr/bin/env python
# ______________________________________________________________________
'''test_indexing

Unit tests for checking Numba's indexing into Numpy arrays.
'''
# ______________________________________________________________________

from numba import double, int_
from numba.decorators import jit

import numpy

import unittest

# ______________________________________________________________________

def get_index_fn_0 (inarr):
    return inarr[1,2,3]

def set_index_fn_0 (ioarr):
    ioarr[1,2,3] = 0.

def set_index_fn_1 (min_x, max_x, min_y, out_arr):
    '''Thinly veiled (and simplified) version of the Mandelbrot
    driver...though this is very similar to just doing a zip of
    arange(min_x,max_x + epsilon,delta)[mgrid[:width,:height][0]] (and
    the corresponding y values).'''
    width = out_arr.shape[0]
    height = out_arr.shape[1]
    delta = (max_x - min_x) / width
    for x in range(width):
        x_val = x * delta + min_x
        for y in range(height):
            y_val = y * delta + min_y
            out_arr[x,y,0] = x_val
            out_arr[x,y,1] = y_val

def set_index_fn_2(arr):
    width = arr.shape[0]
    height = arr.shape[1]
    for x in range(width):
        for y in range(height):
            arr[x, y] = x*width+y

def get_shape_fn_0 (arr):
    width = arr.shape[0]
    return width

def get_shape_fn_1 (arr):
    height = arr.shape[1]
    return height

def get_shape_fn_2 (arr):
    height = arr.shape[2]
    return height

# ______________________________________________________________________

class TestIndexing (unittest.TestCase):
    def test_get_index_fn_0 (self):
        arr = numpy.ones((4,4,4), dtype=numpy.double)
        arr[1,2,3] = 0.
        compiled_fn = jit(restype=double,
                                    argtypes=[double[:, :, ::1]])(get_index_fn_0)
        self.assertEqual(compiled_fn(arr), 0.)

    def test_set_index_fn_0 (self):
        arr = numpy.ones((4,4,4))
        compiled_fn = jit(argtypes=[double[:,:,::1]])(set_index_fn_0)
        self.assertEqual(arr[1,2,3], 1.)
        compiled_fn(arr)
        self.assertEqual(arr[1,2,3], 0.)

    def test_set_index_fn_1 (self):
        control_arr = numpy.zeros((50, 50, 2), dtype=numpy.double)
        test_arr = numpy.zeros_like(control_arr)
        set_index_fn_1(-1., 1., -1., control_arr)
        argtypes = double, double, double, double[:,:,:]
        compiled_fn = jit(argtypes=argtypes)(set_index_fn_1)
        compiled_fn(-1., 1., -1., test_arr)
        self.assertTrue((numpy.abs(control_arr - test_arr) < 1e9).all())

    def test_get_shape_fn_0(self):
        arr = numpy.zeros((5,6,7), dtype=numpy.double)
        compiled_fn = jit(restype=int_,
                                    argtypes=[double[:, :, ::1]])(get_shape_fn_0)
        self.assertEqual(compiled_fn(arr), 5)

    def test_get_shape_fn_1(self):
        arr = numpy.zeros((5,6,7), dtype=numpy.double)
        compiled_fn = jit(restype=int_,
                                    argtypes=[double[:, :, ::1]])(get_shape_fn_1)
        self.assertEqual(compiled_fn(arr), 6)

    def test_get_shape_fn_2(self):
        arr = numpy.zeros((5,6,7), dtype=numpy.double)
        compiled_fn = jit(restype=int_,
                                    argtypes=[double[:, :, ::1]])(get_shape_fn_2)
        self.assertEqual(compiled_fn(arr), 7)

    def test_set_index_fn_2 (self):
        control_arr = numpy.zeros((10, 10), dtype=numpy.double)
        test_arr = numpy.zeros_like(control_arr)

        set_index_fn_2(control_arr)

        argtypes = double[:, :],
        compiled_fn = jit(argtypes=argtypes)(set_index_fn_2)

        compiled_fn(test_arr)

        self.assertTrue((numpy.abs(control_arr - test_arr) < 1e9).all())



# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main()

# ______________________________________________________________________
# End of test_indexing.py
