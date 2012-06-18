#! /usr/bin/env python
# ______________________________________________________________________
'''test_indexing

Unit tests for checking Numba's indexing into Numpy arrays.
'''
# ______________________________________________________________________

from numba.decorators import numba_compile

import numpy

import unittest

# ______________________________________________________________________

def get_index_fn_0 (inarr):
    return inarr[1,2,3]

def set_index_fn_0 (ioarr):
    ioarr[1,2,3] = 0.

# ______________________________________________________________________

class TestIndexing (unittest.TestCase):
    def test_get_index_fn_0 (self):
        arr = numpy.ones((4,4,4))
        arr[1,2,3] = 0.
        compiled_fn = numba_compile(arg_types = [['d']])(get_index_fn_0)
        self.assertEqual(compiled_fn(arr), 0.)

    def test_set_index_fn_0 (self):
        arr = numpy.ones((4,4,4))
        compiled_fn = numba_compile(arg_types = [['d']])(set_index_fn_0)
        self.assertEqual(arr[1,2,3], 1.)
        compiled_fn(arr)
        self.assertEqual(arr[1,2,3], 0.)

# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main()

# ______________________________________________________________________
# End of test_indexing.py
