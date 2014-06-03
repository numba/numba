from __future__ import print_function, absolute_import, division
import numpy
from numbapro import cuda
from numbapro.testsupport import unittest


def cuadd(a, b):
    i = cuda.grid(1)
    a[i] += b[i]


class TestArithErrors(unittest.TestCase):
    @unittest.expectedFailure
    def test_signed_overflow(self):
        jitted = cuda.jit('void(int8[:], int8[:])', debug=True)(cuadd)
        a = numpy.array(list(reversed(range(128))), dtype=numpy.int8)

        try:
            jitted[1, a.size](a, a)
        except RuntimeError as e:
            i = e.tid[0]
            self.assertTrue(int(a[i]) + int(a[i]) > 127)
        else:
            raise AssertionError('expecting an exception')

    @unittest.expectedFailure
    def test_signed_overflow2(self):
        jitted = cuda.jit('void(int8[:], int8[:])', debug=True)(cuadd)
        a = numpy.array(list(range(128)), dtype=numpy.int8)

        try:
            jitted[1, a.size](a, a)
        except RuntimeError as e:
            i = e.tid[0]
            self.assertTrue(int(a[i]) + int(a[i]) > 127)
        else:
            raise AssertionError('expecting an exception')
        
    @unittest.expectedFailure
    def test_signed_overflow3(self):
        jitted = cuda.jit('void(int8[:], int8[:])', debug=True)(cuadd)
        a = numpy.array(list(range(128)), dtype=numpy.int8)

        try:
            jitted[2, a.size // 2](a, a)
        except RuntimeError as e:
            i = e.tid[0] + e.ctaid[0] * a.size // 2
            self.assertTrue(int(a[i]) + int(a[i]) > 127)
        else:
            raise AssertionError('expecting an exception')


if __name__ == '__main__':
    unittest.main()
