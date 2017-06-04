from __future__ import print_function

import numpy as np

from numba import ocl
from numba import unittest_support as unittest
from numba.ocl.testing import captured_ocl_stdout


def cuhello():
    i = ocl.grid(1)
    print(i, 999)
    print(-42)


def printfloat():
    i = ocl.grid(1)
    print(i, 23, 34.75, 321)


def printstring():
    i = ocl.grid(1)
    print(i, "hop!", 999)


def printempty():
    print()


class TestPrint(unittest.TestCase):

    def test_cuhello(self):
        jcuhello = ocl.jit('void()', debug=False)(cuhello)
        with captured_ocl_stdout() as stdout:
            jcuhello[2, 3]()
        # The output of GPU threads is intermingled, but each print()
        # call is still atomic
        out = stdout.getvalue()
        lines = sorted(out.splitlines(True))
        expected = ['-42\n'] * 6 + ['%d 999\n' % i for i in range(6)]
        self.assertEqual(lines, expected)

    def test_printfloat(self):
        jprintfloat = ocl.jit('void()', debug=False)(printfloat)
        with captured_ocl_stdout() as stdout:
            jprintfloat()
        # OpenCL and the simulator use different formats for float formatting
        self.assertIn(stdout.getvalue(), ["0 23 34.750000 321\n",
                                          "0 23 34.75 321\n"])

    def test_printempty(self):
        cufunc = ocl.jit('void()', debug=False)(printempty)
        with captured_ocl_stdout() as stdout:
            cufunc()
        self.assertEqual(stdout.getvalue(), "\n")

    def test_string(self):
        cufunc = ocl.jit('void()', debug=False)(printstring)
        with captured_ocl_stdout() as stdout:
            cufunc[1, 3]()
        out = stdout.getvalue()
        lines = sorted(out.splitlines(True))
        expected = ['%d hop! 999\n' % i for i in range(3)]
        self.assertEqual(lines, expected)


if __name__ == '__main__':
    unittest.main()
