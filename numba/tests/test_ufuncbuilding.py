from __future__ import print_function, absolute_import, division
import numpy
from numba import unittest_support as unittest
from numba.npyufunc import UFuncBuilder


def add(a, b):
    return a + b


class TestUfuncBuilding(unittest.TestCase):
    def test_basic_ufunc(self):
        ufb = UFuncBuilder(add)
        ufb.add("int32(int32, int32)")
        ufb.add("int64(int64, int64)")
        ufunc = ufb.build_ufunc()

        a = numpy.arange(10, dtype='int32')
        b = ufunc(a, a)
        self.assertTrue(numpy.all(a + a == b))

    def test_ufunc_struct(self):
        ufb = UFuncBuilder(add)
        ufb.add("complex64(complex64, complex64)")
        ufunc = ufb.build_ufunc()

        a = numpy.arange(10, dtype='complex64') + 1j
        b = ufunc(a, a)
        self.assertTrue(numpy.all(a + a == b))


if __name__ == '__main__':
    unittest.main()
