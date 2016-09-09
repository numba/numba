#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import


from timeit import default_timer as timer

from matplotlib.pylab import imshow, jet, show, ion

from numba import vectorize
import numpy as np

sig = 'uint8(uint32, f4, f4, f4, f4, uint32, uint32, uint32)'

@vectorize([sig], target='cuda')
def mandel(tid, min_x, max_x, min_y, max_y, width, height, iters):
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    x = tid % width
    y = tid / width

    real = min_x + x * pixel_size_x
    imag = min_y + y * pixel_size_y

    c = complex(real, imag)
    z = 0.0j

    for i in range(iters):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            return i
    return 255

def create_fractal(min_x, max_x, min_y, max_y, width, height, iters):
    tids = np.arange(width * height, dtype=np.uint32)
    return mandel(tids, np.float32(min_x), np.float32(max_x), np.float32(min_y),
                  np.float32(max_y), np.uint32(height), np.uint32(width),
                  np.uint32(iters))

def main():
    width = 500 * 10
    height = 750 * 10
    ts = timer()
    pixels = create_fractal(-2.0, 1.0, -1.0, 1.0, width, height, 20)
    te = timer()
    print('time: %f' % (te - ts))
    image = pixels.reshape(width, height)
    #print(image)
    imshow(image)
    show()


if __name__ == '__main__':
    main()
