#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from scipy.misc import lena
from numpy import ones
import numpy

from numba.decorators import jit
from numba import int32, int64

# Original approach will be slower for now due to the object mode failback
# for numpy.zero_like
#
# @jit(argtypes=[int32[:,:], int32[:,:]], restype=int32[:,:])
# def filter2d(image, filt):
#     M, N = image.shape
#     Mf, Nf = filt.shape
#     Mf2 = Mf // 2
#     Nf2 = Nf // 2
#     result = numpy.zeros_like(image)
#     for i in range(Mf2, M - Mf2):
#         for j in range(Nf2, N - Nf2):
#             num = 0.0
#             for ii in range(Mf):
#                 for jj in range(Nf):
#                     num += (filt[Mf-1-ii, Nf-1-jj] * image[i-Mf2+ii, j-Nf2+jj])
#             result[i, j] = num
#     return result


@jit(nopython=True)
def filter2d_core(image, filt, result):
    M, N = image.shape
    Mf, Nf = filt.shape
    Mf2 = Mf // 2
    Nf2 = Nf // 2
    for i in range(Mf2, M - Mf2):
        for j in range(Nf2, N - Nf2):
            num = 0
            for ii in range(Mf):
                for jj in range(Nf):
                    num += (filt[Mf-1-ii, Nf-1-jj] * image[i-Mf2+ii,j-Nf2+jj])
            result[i, j] = num

@jit
def filter2d(image, filt):
    result = numpy.zeros_like(image)
    filter2d_core(image, filt, result)
    return result


image = lena()
filter = ones((7,7), dtype='int32')

result = filter2d(image, filter)   # warm up

import time
start = time.time()
result = filter2d(image, filter)
duration = time.time() - start

from scipy.ndimage import convolve
start = time.time()
result2 = convolve(image, filter)
duration2 = time.time() - start

print("Time for Numba filter = %f\nTime for scipy convolve = %f" % (duration, duration2))

from pylab import subplot, imshow, show, title, gray
subplot(1,2,1)
imshow(image)
title('Original Image')
gray()
subplot(1,2,2)
imshow(result)
title('Filtered Image')
gray()
show()
