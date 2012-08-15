#! /usr/bin/env python
# ______________________________________________________________________
'''test_filter2d

Test the filter2d() example from the PyCon'12 slide deck.
'''
# ______________________________________________________________________

import numpy

from numba import *
from numba.decorators import jit

import sys
import unittest

# ______________________________________________________________________

def sum2d(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    return result

# ______________________________________________________________________

class TestSum2d(unittest.TestCase):
    def test_vectorized_sum2d(self):
        usum2d = jit(arg_types=[double[:,:]],
                     ret_type=double)(sum2d)
        image = numpy.random.rand(10, 10)
        plain_old_result = sum2d(image)
        hot_new_result = usum2d(image)
        self.assertTrue((abs(plain_old_result - hot_new_result) < 1e-9))

# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main(*sys.argv[1:])

# ______________________________________________________________________
# End of test_filter2d.py
