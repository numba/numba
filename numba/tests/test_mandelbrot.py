from __future__ import print_function

import numba.unittest_support as unittest
from numba.compiler import compile_isolated, Flags
from numba import types, utils

enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

force_pyobj_flags = Flags()
force_pyobj_flags.set("force_pyobject")


def is_in_mandelbrot(c):
    i = 0
    z = 0.0j
    for i in range(100):
        z = z ** 2 + c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            return False
    return True


class TestMandelbrot(unittest.TestCase):

    def test_mandelbrot(self):
        pyfunc = is_in_mandelbrot
        cr = compile_isolated(pyfunc, (types.complex64,))
        cfunc = cr.entry_point

        points = [0+0j, 1+0j, 0+1j, 1+1j, 0.1+0.1j]
        for p in points:
            self.assertEqual(cfunc(p), pyfunc(p))


if __name__ == '__main__':
    unittest.main()
