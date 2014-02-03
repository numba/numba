from __future__ import print_function, absolute_import, division
import numpy
from numba import unittest_support as unittest
from numba.npyufunc.ufuncbuilder import UFuncBuilder, GUFuncBuilder
from numba import vectorize, guvectorize

def add(a, b):
    return a + b


def guadd(a, b, c):
    x, y = c.shape
    for i in range(x):
        for j in range(y):
            c[i, j] = a[i, j] + b[i, j]


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


class TestGUfuncBuilding(unittest.TestCase):
    def test_basic_gufunc(self):
        gufb = GUFuncBuilder(guadd, "(x, y),(x, y)->(x, y)")
        gufb.add("void(int32[:,:], int32[:,:], int32[:,:])")
        ufunc = gufb.build_ufunc()

        a = numpy.arange(10, dtype="int32").reshape(2, 5)
        b = ufunc(a, a)

        self.assertTrue(numpy.all(a + a == b))

    def test_gufunc_struct(self):
        gufb = GUFuncBuilder(guadd, "(x, y),(x, y)->(x, y)")
        gufb.add("void(complex64[:,:], complex64[:,:], complex64[:,:])")
        ufunc = gufb.build_ufunc()

        a = numpy.arange(10, dtype="complex64").reshape(2, 5) + 1j
        b = ufunc(a, a)

        self.assertTrue(numpy.all(a + a == b))


class TestVectorizeDecor(unittest.TestCase):
    def test_vectorize(self):
        ufunc = vectorize(['int32(int32, int32)'])(add)
        a = numpy.arange(10, dtype='int32')
        b = ufunc(a, a)
        self.assertTrue(numpy.all(a + a == b))

    def test_guvectorize(self):
        ufunc = guvectorize(['(int32[:,:], int32[:,:], int32[:,:])'],
                            "(x,y),(x,y)->(x,y)")(guadd)
        a = numpy.arange(10, dtype='int32').reshape(2, 5)
        b = ufunc(a, a)
        self.assertTrue(numpy.all(a + a == b))

if __name__ == '__main__':
    unittest.main()
