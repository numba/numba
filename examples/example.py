
from scipy.misc import lena
from numpy import ones
import numpy

from numba.decorators import jit
from numba import int32

@jit(arg_types=[int32[:,:], int32[:,:]], ret_type=int32[:,:])
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

image = lena()
filter = ones((7,7), dtype='int32')
result = filter2d(image, filter)

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
