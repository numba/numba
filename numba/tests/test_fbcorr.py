#! /usr/bin/env python
# ______________________________________________________________________
'''test_fbcorr

Test the fbcorr() example .... 
'''
# ______________________________________________________________________

import numpy as np
import numba
from numba.decorators import jit
nd4type = numba.double[:,:,:,:]

import sys
import unittest

# ______________________________________________________________________

def fbcorr(imgs, filters, output):
    n_imgs, n_rows, n_cols, n_channels = imgs.shape
    n_filters, height, width, n_ch2 = filters.shape

    for ii in range(n_imgs):
        for rr in range(n_rows - height + 1):
            for cc in range(n_cols - width + 1):
                for hh in xrange(height):
                    for ww in xrange(width):
                        for jj in range(n_channels):
                            for ff in range(n_filters):
                                imgval = imgs[ii, rr + hh, cc + ww, jj]
                                filterval = filters[ff, hh, ww, jj]
                                output[ii, ff, rr, cc] += imgval * filterval

# ______________________________________________________________________

class TestFbcorr(unittest.TestCase):
    def test_vectorized_fbcorr(self):
        ufbcorr = jit(argtypes=(nd4type, nd4type, nd4type))(fbcorr)

        imgs = np.random.randn(10, 16, 16, 3)
        filt = np.random.randn(6, 5, 5, 3)
        old_output = np.zeros((10, 6, 15, 15))
        fbcorr(imgs, filt, old_output)
        new_output = np.zeros((10, 6, 15, 15))
        ufbcorr(imgs, filt, new_output)

        self.assertTrue((abs(old_output - new_output) < 1e-9).all())

    def test_catch_error(self):
        imgs = np.random.randn(10, 64, 64, 3)
        filt = np.random.randn(6, 5, 5, 3)
        #incorrect channel-minor format?
        old_output = np.zeros((10, 60, 60, 6))

        try:
            fbcorr(imgs, filt, old_output)
        except IndexError, e:
            print 'This test produced the error "' + repr(e) + '"'
        else:
            raise Exception('This should have produced an error.')

# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main(*sys.argv[1:])

# ______________________________________________________________________
# End of test_filter2d.py
