#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from scipy.misc import ascent
from numpy import ones
import numpy

from numba.decorators import jit


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


@jit(nopython=True)
def filter2d(image, filt):
    result = numpy.zeros_like(image)
    filter2d_core(image, filt, result)
    return result


image = ascent().astype(numpy.float64)
filter = ones((7,7), dtype=image.dtype)

result = filter2d(image, filter)   # warm up

from timeit import default_timer as time

start = time()
result = filter2d(image, filter)
duration = time() - start

from scipy.ndimage import convolve

start = time()
result2 = convolve(image, filter)
duration2 = time() - start

print("Time for Numba filter = %f\nTime for scipy convolve = %f" % (duration, duration2))

from pylab import subplot, imshow, show, title, gray

subplot(1,3,1)
imshow(image)
title('Original Image')
gray()
subplot(1,3,2)
imshow(result)
title('Numba Filtered Image')
gray()
subplot(1,3,3)
imshow(result2)
title('Scipy Filtered Image')
gray()

show()
