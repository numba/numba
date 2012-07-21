"""
This file demonstrates a filterbank correlation loop.
"""
import numpy as np

from numba.decorators import numba_compile
nd4type = [[[['d']]]]

@numba_compile(ret_type=nd4type, arg_types=(nd4type, nd4type, nd4type))
def fbcorr(imgs, filters, output):
    n_imgs, n_rows, n_cols, n_channels = imgs.shape
    n_filters, height, width, n_ch2 = filters.shape

    #output = np.zeros((n_imgs, n_rows - height + 1, n_cols - width + 1, n_filters))
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

    return output

imgs = np.random.randn(10, 64, 64, 3)
filt = np.random.randn(6, 5, 5, 3)
output = np.zeros((10, 60, 60, 6))

import time
t0 = time.time()
fbcorr(imgs, filt, output)
print time.time() - t0
