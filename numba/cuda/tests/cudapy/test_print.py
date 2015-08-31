from __future__ import print_function
import numpy
from numba import cuda
from numba import unittest_support as unittest


def cuhello():
    i = cuda.grid(1)
    print(i, 1.234)


def printfloat():
    i = cuda.grid(1)
    print(i, 23, 34.3, 321)


def cuprintary(A):
    i = cuda.grid(1)
    print("A[", i, "]", A[i])


class TestPrint(unittest.TestCase):
    def test_cuhello(self):
        """
        Eyeballing required
        """
        jcuhello = cuda.jit('void()', debug=False)(cuhello)
        jcuhello[2, 3]()
        cuda.synchronize()

    def test_printfloat(self):
        """
        Eyeballing required
        """
        jprintfloat = cuda.jit('void()', debug=False)(printfloat)
        jprintfloat()
        cuda.synchronize()

    @unittest.skipIf(True, "Print string not implemented yet")
    def test_print_array(self):
        """
        Eyeballing required
        """
        jcuprintary = cuda.jit('void(float32[:])')(cuprintary)
        A = numpy.arange(10, dtype=numpy.float32)
        jcuprintary[2, 5](A)
        cuda.synchronize()


if __name__ == '__main__':
    unittest.main()
