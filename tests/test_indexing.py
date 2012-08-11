#! /usr/bin/env python
# ______________________________________________________________________
'''test_indexing

Unit tests for checking Numba's indexing into Numpy arrays.
'''
# ______________________________________________________________________

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

# ______________________________________________________________________

class TestIndexing (unittest.TestCase):
    def test_get_index_fn_0 (self):
        arr = numpy.ones((4,4,4))
        arr[1,2,3] = 0.
        compiled_fn = jit(arg_types = [['d']])(get_index_fn_0)
        self.assertEqual(compiled_fn(arr), 0.)

    def test_set_index_fn_0 (self):
        arr = numpy.ones((4,4,4))
        compiled_fn = jit(arg_types = [['d']])(set_index_fn_0)
        self.assertEqual(arr[1,2,3], 1.)
        compiled_fn(arr)
        self.assertEqual(arr[1,2,3], 0.)

    def test_set_index_fn_1 (self):
        control_arr = numpy.zeros((50, 50, 2))
        test_arr = numpy.zeros_like(control_arr)
        set_index_fn_1(-1., 1., -1., control_arr)
        compiled_fn = jit(
            arg_types = ['d', 'd', 'd', ['d']])(set_index_fn_1)
        compiled_fn(-1., 1., -1., test_arr)
        self.assertTrue((numpy.abs(control_arr - test_arr) < 1e9).all())

# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main()

# ______________________________________________________________________
# End of test_indexing.py
