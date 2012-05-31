#! /usr/bin/env python
# ______________________________________________________________________
'''test_mandelbrot

Test the Numba compiler on several variants of Mandelbrot set membership
computations.
'''
# ______________________________________________________________________

from numba.decorators import numba_compile

import unittest

# ______________________________________________________________________

#@numba_compile(arg_types = ['d','d','i'], ret_type = 'i')
def mandel_1(real_coord, imag_coord, max_iters):
    '''Given a the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the Mandelbrot
    set given a fixed number of iterations.
    Inspired by code at http://wiki.cython.org/examples/mandelbrot
    '''
    # Ideally we'd want to use a for loop, but we'll need to be able
    # to detect and desugar for loops over range/xrange/arange first.
    i = 0
    z_real = 0.
    z_imag = 0.
    while i < max_iters:
        z_real = z_real * z_real - z_imag * z_imag + real_coord
        z_imag = 2 * z_real * z_imag + imag_coord
        if (z_real * z_real + z_imag * z_imag) >= 4:
            return i
        i += 1
    return -1

# ______________________________________________________________________

class TestMandelbrot(unittest.TestCase):
    def test_mandel_1_sanity(self):
        mandel_1c = numba_compile(arg_types = ['d', 'd', 'i'],
                                  ret_type = 'i')(mandel_1)
        self.assertEquals(mandel_1c(0., 0., 20), -1)

# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main()

# ______________________________________________________________________
# End of test_mandelbrot.py
