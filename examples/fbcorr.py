#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file demonstrates a filterbank correlation loop.
"""
from __future__ import print_function, division, absolute_import

import numpy as np

from numba import jit


@jit(nopython=True)
def fbcorr(imgs, filters, output):
    n_imgs, n_rows, n_cols, n_channels = imgs.shape
    n_filters, height, width, n_ch2 = filters.shape

    for ii in range(n_imgs):
        for rr in range(n_rows - height + 1):
            for cc in range(n_cols - width + 1):
                for hh in range(height):
                    for ww in range(width):
                        for jj in range(n_channels):
                            for ff in range(n_filters):
                                imgval = imgs[ii, rr + hh, cc + ww, jj]
                                filterval = filters[ff, hh, ww, jj]
                                output[ii, ff, rr, cc] += imgval * filterval

def main ():
    imgs = np.random.randn(10, 16, 16, 3)
    filt = np.random.randn(6, 5, 5, 3)
    output = np.zeros((10, 6, 15, 15))

    import time
    t0 = time.time()
    fbcorr(imgs, filt, output)
    print(time.time() - t0)

if __name__ == "__main__":
    main()
