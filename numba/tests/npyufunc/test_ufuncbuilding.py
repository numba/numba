import sys

import numpy as np

from numba.np.ufunc.ufuncbuilder import GUFuncBuilder
from numba import vectorize, guvectorize
from numba.np.ufunc import PyUFunc_One
from numba.np.ufunc.dufunc import DUFunc as UFuncBuilder
from numba.tests.support import tag, TestCase
from numba.core import config
import unittest


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

@vectorize(nopython=True)
def inner(a, b):
    return a + b

@vectorize(["int64(int64, int64)"], nopython=True)
def inner_explicit(a, b):
    return a + b

def outer(a, b):
    return inner(a, b)

def outer_explicit(a, b):
    return inner_explicit(a, b)


class Dummy: pass


def guadd_obj(a, b, c):
    Dummy()  # to force object mode
    x, y = c.shape
    for i in range(x):
        for j in range(y):
            c[i, j] = a[i, j] + b[i, j]

def guadd_scalar_obj(a, b, c):
    Dummy()  # to force object mode
    x, y = c.shape
    for i in range(x):
        for j in range(y):
            c[i, j] = a[i, j] + b


class MyException(Exception):
    pass


def guerror(a, b, c):
    raise MyException


class TestUfuncBuilding(TestCase):

    def test_basic_ufunc(self):
        ufb = UFuncBuilder(add)
        cres = ufb.add("int32(int32, int32)")
        self.assertFalse(cres.objectmode)
        cres = ufb.add("int64(int64, int64)")
        self.assertFalse(cres.objectmode)
        ufunc = ufb.build_ufunc()

        def check(a):
            b = ufunc(a, a)
            self.assertPreciseEqual(a + a, b)
            self.assertEqual(b.dtype, a.dtype)

        a = np.arange(12, dtype='int32')
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
            self.assertPreciseEqual(a + a, b)
            self.assertEqual(b.dtype, a.dtype)

        a = np.arange(12, dtype='complex64') + 1j
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

        a = np.arange(10, dtype='int32')
        b = ufunc(a, a)
        self.assertPreciseEqual(a + a, b)

    def test_nested_call(self):
        """
        Check nested call to an implicitly-typed ufunc.
        """
        builder = UFuncBuilder(outer,
                               targetoptions={'nopython': True})
        builder.add("(int64, int64)")
        ufunc = builder.build_ufunc()
        self.assertEqual(ufunc(-1, 3), 2)

    def test_nested_call_explicit(self):
        """
        Check nested call to an explicitly-typed ufunc.
        """
        builder = UFuncBuilder(outer_explicit,
                               targetoptions={'nopython': True})
        builder.add("(int64, int64)")
        ufunc = builder.build_ufunc()
        self.assertEqual(ufunc(-1, 3), 2)


class TestUfuncBuildingJitDisabled(TestUfuncBuilding):

    def setUp(self):
        self.old_disable_jit = config.DISABLE_JIT
        config.DISABLE_JIT = False

    def tearDown(self):
        config.DISABLE_JIT = self.old_disable_jit


class TestGUfuncBuilding(TestCase):

    def test_basic_gufunc(self):
        gufb = GUFuncBuilder(guadd, "(x, y),(x, y)->(x, y)")
        cres = gufb.add("void(int32[:,:], int32[:,:], int32[:,:])")
        self.assertFalse(cres.objectmode)
        ufunc = gufb.build_ufunc()

        a = np.arange(10, dtype="int32").reshape(2, 5)
        b = ufunc(a, a)

        self.assertPreciseEqual(a + a, b)
        self.assertEqual(b.dtype, np.dtype('int32'))

        # Metadata
        self.assertEqual(ufunc.__name__, "guadd")
        self.assertIn("A generalized addition", ufunc.__doc__)

    def test_gufunc_struct(self):
        gufb = GUFuncBuilder(guadd, "(x, y),(x, y)->(x, y)")
        cres = gufb.add("void(complex64[:,:], complex64[:,:], complex64[:,:])")
        self.assertFalse(cres.objectmode)
        ufunc = gufb.build_ufunc()

        a = np.arange(10, dtype="complex64").reshape(2, 5) + 1j
        b = ufunc(a, a)

        self.assertPreciseEqual(a + a, b)

    def test_gufunc_struct_forceobj(self):
        gufb = GUFuncBuilder(guadd, "(x, y),(x, y)->(x, y)",
                             targetoptions=dict(forceobj=True))
        cres = gufb.add("void(complex64[:,:], complex64[:,:], complex64[:,"
                        ":])")
        self.assertTrue(cres.objectmode)
        ufunc = gufb.build_ufunc()

        a = np.arange(10, dtype="complex64").reshape(2, 5) + 1j
        b = ufunc(a, a)

        self.assertPreciseEqual(a + a, b)


class TestGUfuncBuildingJitDisabled(TestGUfuncBuilding):

    def setUp(self):
        self.old_disable_jit = config.DISABLE_JIT
        config.DISABLE_JIT = False

    def tearDown(self):
        config.DISABLE_JIT = self.old_disable_jit


class TestVectorizeDecor(TestCase):

    _supported_identities = [0, 1, None, "reorderable"]

    def test_vectorize(self):
        ufunc = vectorize(['int32(int32, int32)'])(add)
        a = np.arange(10, dtype='int32')
        b = ufunc(a, a)
        self.assertPreciseEqual(a + a, b)

    def test_vectorize_objmode(self):
        ufunc = vectorize(['int32(int32, int32)'], forceobj=True)(add)
        a = np.arange(10, dtype='int32')
        b = ufunc(a, a)
        self.assertPreciseEqual(a + a, b)

    def test_vectorize_bool_return(self):
        ufunc = vectorize(['bool_(int32, int32)'])(equals)
        a = np.arange(10, dtype='int32')
        r = ufunc(a,a)
        self.assertPreciseEqual(r, np.ones(r.shape, dtype=np.bool_))

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
        a = np.linspace(0,1,10)
        b = np.linspace(1,2,10)
        ufunc = vectorize(add)
        self.assertPreciseEqual(ufunc(a,b), a + b)
        ufunc2 = vectorize(add)
        c = np.empty(10)
        ufunc2(a, b, c)
        self.assertPreciseEqual(c, a + b)

    def test_vectorize_only_kws(self):
        a = np.linspace(0,1,10)
        b = np.linspace(1,2,10)
        ufunc = vectorize(identity=PyUFunc_One, nopython=True)(mul)
        self.assertPreciseEqual(ufunc(a,b), a * b)

    def test_vectorize_output_kwarg(self):
        """
        Passing the output array as a keyword argument (issue #1867).
        """
        def check(ufunc):
            a = np.arange(10, 16, dtype='int32')
            out = np.zeros_like(a)
            got = ufunc(a, a, out=out)
            self.assertIs(got, out)
            self.assertPreciseEqual(out, a + a)
            with self.assertRaises(TypeError):
                ufunc(a, a, zzz=out)

        # With explicit sigs
        ufunc = vectorize(['int32(int32, int32)'], nopython=True)(add)
        check(ufunc)
        # With implicit sig
        ufunc = vectorize(nopython=True)(add)
        check(ufunc)  # compiling
        check(ufunc)  # after compiling

    def test_guvectorize(self):
        ufunc = guvectorize(['(int32[:,:], int32[:,:], int32[:,:])'],
                            "(x,y),(x,y)->(x,y)")(guadd)
        a = np.arange(10, dtype='int32').reshape(2, 5)
        b = ufunc(a, a)
        self.assertPreciseEqual(a + a, b)

    def test_guvectorize_no_output(self):
        ufunc = guvectorize(['(int32[:,:], int32[:,:], int32[:,:])'],
                            "(x,y),(x,y),(x,y)")(guadd)
        a = np.arange(10, dtype='int32').reshape(2, 5)
        out = np.zeros_like(a)
        ufunc(a, a, out)
        self.assertPreciseEqual(a + a, out)

    def test_guvectorize_objectmode(self):
        ufunc = guvectorize(['(int32[:,:], int32[:,:], int32[:,:])'],
                            "(x,y),(x,y)->(x,y)")(guadd_obj)
        a = np.arange(10, dtype='int32').reshape(2, 5)
        b = ufunc(a, a)
        self.assertPreciseEqual(a + a, b)

    def test_guvectorize_scalar_objectmode(self):
        """
        Test passing of scalars to object mode gufuncs.
        """
        ufunc = guvectorize(['(int32[:,:], int32, int32[:,:])'],
                            "(x,y),()->(x,y)")(guadd_scalar_obj)
        a = np.arange(10, dtype='int32').reshape(2, 5)
        b = ufunc(a, 3)
        self.assertPreciseEqual(a + 3, b)

    def test_guvectorize_error_in_objectmode(self):
        ufunc = guvectorize(['(int32[:,:], int32[:,:], int32[:,:])'],
                            "(x,y),(x,y)->(x,y)", forceobj=True)(guerror)
        a = np.arange(10, dtype='int32').reshape(2, 5)
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


class TestNEP13WithoutSignature(TestCase):

    def test_all(self):

        # note: no signatures specified
        @vectorize(nopython=True)
        def new_ufunc(hundreds, tens, ones):
            return 100*hundreds + 10*tens + ones

        # https://numpy.org/neps/nep-0013-ufunc-overrides.html
        class NEP13Array:
            def __init__(self, array):
                self.array = array

            def __array__(self):
                return self.array

            def tolist(self):
                return self.array.tolist()

            def __array_ufunc__(self, ufunc, method, *args, **kwargs):
                if method != "__call__":
                    return NotImplemented

                return NEP13Array(ufunc(*[np.asarray(x) for x in args], **kwargs))

        # give it integers
        a = np.array([1, 2, 3], dtype=np.int64)
        b = np.array([4, 5, 6], dtype=np.int64)
        c = np.array([7, 8, 9], dtype=np.int64)

        all_np = new_ufunc(a, b, c)
        self.assertIsInstance(all_np, np.ndarray)
        self.assertEqual(all_np.tolist(), [147, 258, 369])

        nep13_1 = new_ufunc(NEP13Array(a), b, c)
        self.assertIsInstance(nep13_1, NEP13Array)
        self.assertEqual(nep13_1.tolist(), [147, 258, 369])

        nep13_2 = new_ufunc(a, NEP13Array(b), c)
        self.assertIsInstance(nep13_2, NEP13Array)
        self.assertEqual(nep13_2.tolist(), [147, 258, 369])

        nep13_3 = new_ufunc(a, b, NEP13Array(c))
        self.assertIsInstance(nep13_3, NEP13Array)
        self.assertEqual(nep13_3.tolist(), [147, 258, 369])

        # give it floats
        a = np.array([1.1, 2.2, 3.3], dtype=np.float64)
        b = np.array([4.4, 5.5, 6.6], dtype=np.float64)
        c = np.array([7.7, 8.8, 9.9], dtype=np.float64)

        all_np = new_ufunc(a, b, c)
        self.assertIsInstance(all_np, np.ndarray)
        self.assertEqual(all_np.tolist(), [161.7, 283.8, 405.9])

        nep13_1 = new_ufunc(NEP13Array(a), b, c)
        self.assertIsInstance(nep13_1, NEP13Array)
        self.assertEqual(nep13_1.tolist(), [161.7, 283.8, 405.9])

        nep13_2 = new_ufunc(a, NEP13Array(b), c)
        self.assertIsInstance(nep13_2, NEP13Array)
        self.assertEqual(nep13_2.tolist(), [161.7, 283.8, 405.9])

        nep13_3 = new_ufunc(a, b, NEP13Array(c))
        self.assertIsInstance(nep13_3, NEP13Array)
        self.assertEqual(nep13_3.tolist(), [161.7, 283.8, 405.9])


class TestVectorizeDecorJitDisabled(TestVectorizeDecor):

    def setUp(self):
        self.old_disable_jit = config.DISABLE_JIT
        config.DISABLE_JIT = False

    def tearDown(self):
        config.DISABLE_JIT = self.old_disable_jit


if __name__ == '__main__':
    unittest.main()
