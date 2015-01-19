from __future__ import print_function, absolute_import, division

import sys

import numpy

from numba import unittest_support as unittest
from numba.npyufunc.ufuncbuilder import UFuncBuilder, GUFuncBuilder
from numba import vectorize, guvectorize
from . import support


def add(a, b):
    """An addition"""
    return a + b

def equals(a, b):
    return a == b

def guadd(a, b, c):
    """A generalized addition"""
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


def raise_inner():
    raise MyException("I'm here")

def uerror(x):
    if x == 2:
        raise_inner()
    return x + 1

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

        # Metadata
        self.assertEqual(ufunc.__name__, "add")
        self.assertIn("An addition", ufunc.__doc__)

    def test_ufunc_struct(self):
        ufb = UFuncBuilder(add)
        cres = ufb.add("complex64(complex64, complex64)")
        self.assertFalse(cres.objectmode)
        ufunc = ufb.build_ufunc()

        a = numpy.arange(10, dtype='complex64') + 1j
        b = ufunc(a, a)
        self.assertTrue(numpy.all(a + a == b))
        self.assertEqual(b.dtype, numpy.dtype('complex64'))

    def test_ufunc_forceobj(self):
        ufb = UFuncBuilder(add, targetoptions={'forceobj': True})
        cres = ufb.add("int32(int32, int32)")
        self.assertTrue(cres.objectmode)
        ufunc = ufb.build_ufunc()

        a = numpy.arange(10, dtype='int32')
        b = ufunc(a, a)
        self.assertTrue(numpy.all(a + a == b))


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

        # Metadata
        self.assertEqual(ufunc.__name__, "guadd")
        self.assertIn("A generalized addition", ufunc.__doc__)

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

    _supported_identities = [0, 1, None]
    if numpy.__version__ >= '1.7':
        _supported_identities.append("reorderable")

    def test_vectorize(self):
        ufunc = vectorize(['int32(int32, int32)'])(add)
        a = numpy.arange(10, dtype='int32')
        b = ufunc(a, a)
        self.assertTrue(numpy.all(a + a == b))
        self.assertEqual(b.dtype, numpy.dtype('int32'))

    def test_vectorize_objmode(self):
        ufunc = vectorize(['int32(int32, int32)'], forceobj=True)(add)
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

    def test_vectorize_error_in_objectmode(self):
        # An exception raised inside an object mode @vectorized function
        # is printed out and ignored.
        ufunc = vectorize(['int32(int32)'], forceobj=True)(uerror)
        a = numpy.arange(4, dtype='int32')
        b = numpy.zeros_like(a)
        with support.captured_stderr() as err:
            ufunc(a, out=b)
        err = err.getvalue()
        if sys.version_info >= (3, 4):
            self.assertIn("Exception ignored in: 'object mode ufunc'", err)
            self.assertIn("MyException: I'm here", err)
        else:
            self.assertRegexpMatches(err, r"Exception [^\n]* in 'object mode ufunc' ignored")
            self.assertIn("I'm here", err)
        self.assertTrue(numpy.all(b == numpy.array([1, 2, 0, 4])))

    def test_vectorize_identity(self):
        sig = 'int32(int32, int32)'
        for identity in self._supported_identities:
            ufunc = vectorize([sig], identity=identity)(add)
            expected = None if identity == 'reorderable' else identity
            self.assertEqual(ufunc.identity, expected)
        # Default value is None
        ufunc = vectorize([sig])(add)
        self.assertIs(ufunc.identity, None)
        # Invalid values
        with self.assertRaises(ValueError):
            vectorize([sig], identity='none')(add)
        with self.assertRaises(ValueError):
            vectorize([sig], identity=2)(add)

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

    def test_guvectorize_identity(self):
        args = (['(int32[:,:], int32[:,:], int32[:,:])'], "(x,y),(x,y)->(x,y)")
        for identity in self._supported_identities:
            ufunc = guvectorize(*args, identity=identity)(guadd)
            expected = None if identity == 'reorderable' else identity
            self.assertEqual(ufunc.identity, expected)
        # Default value is None
        ufunc = guvectorize(*args)(guadd)
        self.assertIs(ufunc.identity, None)
        # Invalid values
        with self.assertRaises(ValueError):
            guvectorize(*args, identity='none')(add)
        with self.assertRaises(ValueError):
            guvectorize(*args, identity=2)(add)


if __name__ == '__main__':
    unittest.main()
