# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import sys
from numbapro import CU
from contextlib import closing
import numpy as np
from timeit import default_timer as timer
from pylab import imshow, jet, show, ion

def mandel(tid, min_x, max_x, min_y, max_y, image, iters):
    height, width = image.shape

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    x = tid % width
    y = tid / width

    real = min_x + x * pixel_size_x
    imag = min_y + y * pixel_size_y

    c = complex(real, imag)
    z = 0.0j
    color = 255
    for i in range(iters):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            color = i
            break

    image[y, x] = i

def create_fractal(cu, min_x, max_x, min_y, max_y, image, iters):
    height, width = image.shape
    cu.enqueue(mandel,
               ntid=width * height,
               args=(min_x, max_x, min_y, max_y, image, iters))
    return image

def main():
    target = sys.argv[1]
    cu = CU(target)
    width = 500 * 40
    height = 750 * 40
    with closing(cu):
        image = np.zeros((width, height), dtype=np.uint8)
        d_image = cu.output(image)
        s = timer()
        create_fractal(cu, -2.0, 1.0, -1.0, 1.0, d_image, 20)
        cu.wait()
        e = timer()
    print(e - s)
#    print(image)
#    imshow(image)
#    show()


if __name__ == '__main__':
    main()
