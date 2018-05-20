from __future__ import print_function, absolute_import, division

import sys

import numpy as np
from xnd import xnd

from numba import config, unittest_support as unittest
from numba.gumath import jit_xnd
from ..support import tag, TestCase


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

@jit_xnd(nopython=True)
def inner(a, b):
    return a + b

@jit_xnd("... * int64, ... * int64 -> ... * int64", nopython=True)
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

    @tag('important')
    def test_basic_ufunc(self):
        ufb = jit_xnd(add)
        _, cres = ufb.get_or_create_kernel(("int32", "int32"))
        self.assertFalse(cres.objectmode)
        _, cres = ufb.get_or_create_kernel(("int64", "int64"))
        self.assertFalse(cres.objectmode)

        def check(a, xnd_a=None):
            b = ufb(xnd_a or a, xnd_a or a)
            np.testing.assert_allclose(a + a, b)

        a = np.arange(12, dtype='int32')
        check(a)
        # Non-contiguous dimension
        a = a[::2]
        check(a, xnd.from_buffer(np.arange(12, dtype='int32'))[::2])
        # disabled till https://github.com/plures/xnd/issues/20
        # a = a.reshape((2, 3))
        # check(a)

        # Metadata
        self.assertEqual(ufb.__name__, "add")
        self.assertIn("An addition", ufb.__doc__)

    def test_ufunc_struct(self):
        ufb = jit_xnd(add)
        _, cres = ufb.get_or_create_kernel(("complex64", "complex64"))
        self.assertFalse(cres.objectmode)

        def check(a, xnd_a=None):
            b = ufb(xnd_a or a, xnd_a or a)
            np.testing.assert_allclose(a + a, b)

        a = np.arange(12, dtype='complex64') + 1j
        check(a, xnd([x + 1j for x in range(12)]))
        # Non-contiguous dimension
        a = a[::2]
        check(a, xnd([x + 1j for x in range(12)])[::2])
        # disabled till https://github.com/plures/xnd/issues/20
        # a = a.reshape((2, 3))
        # check(a)

    def test_ufunc_forceobj(self):
        ufb = jit_xnd(add, forceobj=True)

        with self.assertRaises(NotImplementedError):
            _, cres = ufb.get_or_create_kernel(("int32", "int32"))
        # self.assertTrue(cres.objectmode)

        # a = np.arange(10, dtype='int32')
        # b = ufb(a, a)
        # np.testing.assert_allclose(a + a, b)

    def test_nested_call(self):
        """
        Check nested call to an implicitly-typed ufunc.
        """
        builder = jit_xnd(outer, nopython=True)
        builder.get_or_create_kernel(("int64", "int64"))
        self.assertEqual(builder(-1, 3).value, 2)

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

    @tag('important')
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

    @tag('important')
    def test_vectorize_bool_return(self):
        ufunc = vectorize(['bool_(int32, int32)'])(equals)
        a = np.arange(10, dtype='int32')
        r = ufunc(a,a)
        self.assertPreciseEqual(r, np.ones(r.shape, dtype=np.bool_))

    @tag('important')
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

    @tag('important')
    def test_guvectorize(self):
        ufunc = guvectorize(['(int32[:,:], int32[:,:], int32[:,:])'],
                            "(x,y),(x,y)->(x,y)")(guadd)
        a = np.arange(10, dtype='int32').reshape(2, 5)
        b = ufunc(a, a)
        self.assertPreciseEqual(a + a, b)

    @tag('important')
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

    @tag('important')
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
