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

def filter2d(image, filt):
    M, N = image.shape
    Mf, Nf = filt.shape
    Mf2 = Mf // 2
    Nf2 = Nf // 2
    result = numpy.zeros_like(image)
    for i in range(Mf2, M - Mf2):
        for j in range(Nf2, N - Nf2):
            num = 0.0
            for ii in range(Mf):
                for jj in range(Nf):
                    num += (filt[Mf-1-ii, Nf-1-jj] * image[i-Mf2+ii, j-Nf2+jj])
            result[i, j] = num
    return result

# ______________________________________________________________________

class TestFilter2d(unittest.TestCase):
    def test_vectorized_filter2d(self):
        ufilter2d = jit(argtypes=[double[:,:], double[:,:]],
                                  restype=double[:,:])(filter2d)
        image = numpy.random.random((50, 50))
        filt = numpy.random.random((5, 5))
        filt /= filt.sum()
        plain_old_result = filter2d(image, filt)
        hot_new_result = ufilter2d(image, filt)
        self.assertTrue((abs(plain_old_result - hot_new_result) < 1e-9).all())

# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main(*sys.argv[1:])

# ______________________________________________________________________
# End of test_filter2d.py
