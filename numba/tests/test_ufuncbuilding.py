from __future__ import print_function, absolute_import, division
import numpy
from numba import unittest_support as unittest
from numba.npyufunc.ufuncbuilder import UFuncBuilder, GUFuncBuilder
from numba import vectorize, guvectorize


def add(a, b):
    return a + b

def equals(a, b):
    return a == b

def guadd(a, b, c):
    x, y = c.shape
    for i in range(x):
        for j in range(y):
            c[i, j] = a[i, j] + b[i, j]


class Dummy: pass


def guadd_obj(a, b, c):
    Dummy()  # to force object mode
    x, y = c.shape
    for i in range(x):
        for j in range(y):
            c[i, j] = a[i, j] + b[i, j]


class MyException(Exception):
    pass


def guerror(a, b, c):
    raise MyException


class TestUfuncBuilding(unittest.TestCase):
    def test_basic_ufunc(self):
        ufb = UFuncBuilder(add)
        cres = ufb.add("int32(int32, int32)")
        self.assertFalse(cres.objectmode)
        cres = ufb.add("int64(int64, int64)")
        self.assertFalse(cres.objectmode)
        ufunc = ufb.build_ufunc()

        a = numpy.arange(10, dtype='int32')
        b = ufunc(a, a)
        self.assertTrue(numpy.all(a + a == b))


    def test_ufunc_struct(self):
        ufb = UFuncBuilder(add)
        cres = ufb.add("complex64(complex64, complex64)")
        self.assertFalse(cres.objectmode)
        ufunc = ufb.build_ufunc()

        a = numpy.arange(10, dtype='complex64') + 1j
        b = ufunc(a, a)
        self.assertTrue(numpy.all(a + a == b))
        self.assertEqual(b.dtype, numpy.dtype('complex64'))


class TestGUfuncBuilding(unittest.TestCase):
    def test_basic_gufunc(self):
        gufb = GUFuncBuilder(guadd, "(x, y),(x, y)->(x, y)")
        cres = gufb.add("void(int32[:,:], int32[:,:], int32[:,:])")
        self.assertFalse(cres.objectmode)
        ufunc = gufb.build_ufunc()

        a = numpy.arange(10, dtype="int32").reshape(2, 5)
        b = ufunc(a, a)

        self.assertTrue(numpy.all(a + a == b))
        self.assertEqual(b.dtype, numpy.dtype('int32'))


    def test_gufunc_struct(self):
        gufb = GUFuncBuilder(guadd, "(x, y),(x, y)->(x, y)")
        cres = gufb.add("void(complex64[:,:], complex64[:,:], complex64[:,:])")
        self.assertFalse(cres.objectmode)
        ufunc = gufb.build_ufunc()

        a = numpy.arange(10, dtype="complex64").reshape(2, 5) + 1j
        b = ufunc(a, a)

        self.assertTrue(numpy.all(a + a == b))

    def test_gufunc_struct_forceobj(self):
        gufb = GUFuncBuilder(guadd, "(x, y),(x, y)->(x, y)",
                             targetoptions=dict(forceobj=True))
        cres = gufb.add("void(complex64[:,:], complex64[:,:], complex64[:,"
                        ":])")
        self.assertTrue(cres.objectmode)
        ufunc = gufb.build_ufunc()

        a = numpy.arange(10, dtype="complex64").reshape(2, 5) + 1j
        b = ufunc(a, a)

        self.assertTrue(numpy.all(a + a == b))
        self.assertEqual(b.dtype, numpy.dtype('complex64'))


class TestVectorizeDecor(unittest.TestCase):
    def test_vectorize(self):
        ufunc = vectorize(['int32(int32, int32)'])(add)
        a = numpy.arange(10, dtype='int32')
        b = ufunc(a, a)
        self.assertTrue(numpy.all(a + a == b))
        self.assertEqual(b.dtype, numpy.dtype('int32'))

    def test_vectorize_bool_return(self):
        ufunc = vectorize(['bool_(int32, int32)'])(equals)
        a = numpy.arange(10, dtype='int32')
        r = ufunc(a,a)
        self.assertTrue(numpy.all(r))
        self.assertEqual(r.dtype, numpy.dtype('bool_'))

    def test_guvectorize(self):
        ufunc = guvectorize(['(int32[:,:], int32[:,:], int32[:,:])'],
                            "(x,y),(x,y)->(x,y)")(guadd)
        a = numpy.arange(10, dtype='int32').reshape(2, 5)
        b = ufunc(a, a)
        self.assertTrue(numpy.all(a + a == b))
        self.assertEqual(b.dtype, numpy.dtype('int32'))

    def test_guvectorize_objectmode(self):
        ufunc = guvectorize(['(int32[:,:], int32[:,:], int32[:,:])'],
                            "(x,y),(x,y)->(x,y)")(guadd_obj)
        a = numpy.arange(10, dtype='int32').reshape(2, 5)
        b = ufunc(a, a)
        self.assertTrue(numpy.all(a + a == b))

    def test_guvectorize_error_in_objectmode(self):
        ufunc = guvectorize(['(int32[:,:], int32[:,:], int32[:,:])'],
                            "(x,y),(x,y)->(x,y)", forceobj=True)(guerror)
        a = numpy.arange(10, dtype='int32').reshape(2, 5)
        with self.assertRaises(MyException):
            ufunc(a, a)


if __name__ == '__main__':
    unittest.main()
