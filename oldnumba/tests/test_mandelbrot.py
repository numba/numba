"""
>>> image = np.zeros((50, 75), dtype=np.uint8)
>>> numpy_image = image.copy()
>>> image = create_fractal(-2.0, 1.0, -1.0, 1.0, image, 20)
>>> numpy_image = create_fractal.py_func(-2.0, 1.0, -1.0, 1.0, numpy_image, 20)
>>> assert np.allclose(image, numpy_image)
"""

from numba import *
import numpy as np

@autojit(nopython=True)
def mandel(x, y, max_iters):
    """
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the Mandelbrot
    set given a fixed number of iterations.
    """
    i = 0
    c = complex(x,y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i

    return 255

@autojit
def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    with nopython:
        height = image.shape[0]
        width = image.shape[1]

        pixel_size_x = (max_x - min_x) / width
        pixel_size_y = (max_y - min_y) / height
        for x in range(width):
            real = min_x + x * pixel_size_x
            for y in range(height):
                imag = min_y + y * pixel_size_y
                color = mandel(real, imag, iters)
                image[y, x] = color

    return image


if __name__ == "__main__":
    import numba
    numba.testing.testmod()
