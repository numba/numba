from __future__ import print_function, absolute_import, division

import sys

import numpy

from numba import config, unittest_support as unittest
from numba.npyufunc.ufuncbuilder import UFuncBuilder, GUFuncBuilder
from numba import vectorize, guvectorize
from numba.npyufunc import PyUFunc_One
from numba.tests import support


def add(a, b):
    """An addition"""
    return a + b

def equals(a, b):
    return a == b

def mul(a, b):
    """A multiplication"""
    return a * b

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

        def check(a):
            b = ufunc(a, a)
            self.assertTrue(numpy.all(a + a == b))
            self.assertEqual(b.dtype, a.dtype)

        a = numpy.arange(12, dtype='int32')
        check(a)
        # Non-contiguous dimension
        a = a[::2]
        check(a)
        a = a.reshape((2, 3))
        check(a)

        # Metadata
        self.assertEqual(ufunc.__name__, "add")
        self.assertIn("An addition", ufunc.__doc__)

    def test_ufunc_struct(self):
        ufb = UFuncBuilder(add)
        cres = ufb.add("complex64(complex64, complex64)")
        self.assertFalse(cres.objectmode)
        ufunc = ufb.build_ufunc()

        def check(a):
            b = ufunc(a, a)
            self.assertTrue(numpy.all(a + a == b))
            self.assertEqual(b.dtype, a.dtype)

        a = numpy.arange(12, dtype='complex64') + 1j
        check(a)
        # Non-contiguous dimension
        a = a[::2]
        check(a)
        a = a.reshape((2, 3))
        check(a)

    def test_ufunc_forceobj(self):
        ufb = UFuncBuilder(add, targetoptions={'forceobj': True})
        cres = ufb.add("int32(int32, int32)")
        self.assertTrue(cres.objectmode)
        ufunc = ufb.build_ufunc()

        a = numpy.arange(10, dtype='int32')
        b = ufunc(a, a)
        self.assertTrue(numpy.all(a + a == b))


class TestUfuncBuildingJitDisabled(TestUfuncBuilding):

    def setUp(self):
        self.old_disable_jit = config.DISABLE_JIT
        config.DISABLE_JIT = False

    def tearDown(self):
        config.DISABLE_JIT = self.old_disable_jit


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


class TestGUfuncBuildingJitDisabled(TestGUfuncBuilding):

    def setUp(self):
        self.old_disable_jit = config.DISABLE_JIT
        config.DISABLE_JIT = False

    def tearDown(self):
        config.DISABLE_JIT = self.old_disable_jit


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

    def test_vectorize_no_args(self):
        a = numpy.linspace(0,1,10)
        b = numpy.linspace(1,2,10)
        ufunc = vectorize(add)
        self.assertTrue(numpy.all(ufunc(a,b) == (a + b)))
        ufunc2 = vectorize(add)
        c = numpy.empty(10)
        ufunc2(a, b, c)
        self.assertTrue(numpy.all(c == (a + b)))

    def test_vectorize_only_kws(self):
        a = numpy.linspace(0,1,10)
        b = numpy.linspace(1,2,10)
        ufunc = vectorize(identity=PyUFunc_One, nopython=True)(mul)
        self.assertTrue(numpy.all(ufunc(a,b) == (a * b)))

    def test_guvectorize(self):
        ufunc = guvectorize(['(int32[:,:], int32[:,:], int32[:,:])'],
                            "(x,y),(x,y)->(x,y)")(guadd)
        a = numpy.arange(10, dtype='int32').reshape(2, 5)
        b = ufunc(a, a)
        self.assertTrue(numpy.all(a + a == b))
        self.assertEqual(b.dtype, numpy.dtype('int32'))

    def test_guvectorize_no_output(self):
        ufunc = guvectorize(['(int32[:,:], int32[:,:], int32[:,:])'],
                            "(x,y),(x,y),(x,y)")(guadd)
        a = numpy.arange(10, dtype='int32').reshape(2, 5)
        out = numpy.zeros_like(a)
        ufunc(a, a, out)
        self.assertTrue(numpy.all(a + a == out))

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

    def test_guvectorize_invalid_layout(self):
        sigs = ['(int32[:,:], int32[:,:], int32[:,:])']
        # Syntax error
        with self.assertRaises(ValueError) as raises:
            guvectorize(sigs, ")-:")(guadd)
        self.assertIn("bad token in signature", str(raises.exception))
        # Output shape can't be inferred from inputs
        with self.assertRaises(NameError) as raises:
            guvectorize(sigs, "(x,y),(x,y)->(x,z,v)")(guadd)
        self.assertEqual(str(raises.exception),
                         "undefined output symbols: v,z")
        # Arrow but no outputs
        with self.assertRaises(ValueError) as raises:
            guvectorize(sigs, "(x,y),(x,y),(x,y)->")(guadd)
        # (error message depends on Numpy version)


class TestVectorizeDecorJitDisabled(TestVectorizeDecor):

    def setUp(self):
        self.old_disable_jit = config.DISABLE_JIT
        config.DISABLE_JIT = False

    def tearDown(self):
        config.DISABLE_JIT = self.old_disable_jit


if __name__ == '__main__':
    unittest.main()
