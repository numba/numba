# -*- coding: utf-8 -*-

"""
This file demonstrates a filterbank correlation loop.
"""
from __future__ import print_function, division, absolute_import
import numpy as np
import numba
from numba.utils import IS_PY3
from numba.decorators import jit
nd4type = numba.double[:,:,:,:]

if IS_PY3:
    xrange = range

@jit(argtypes=(nd4type, nd4type, nd4type))
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

def main ():
    imgs = np.random.randn(10, 64, 64, 3)
    filt = np.random.randn(6, 5, 5, 3)
    output = np.zeros((10, 60, 60, 6))

    import time
    t0 = time.time()
    fbcorr(imgs, filt, output)
    print(time.time() - t0)

if __name__ == "__main__":
    main()
