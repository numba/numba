from __future__ import print_function, division, absolute_import

import numpy as np

from numba import unittest_support as unittest
from numba import jit, types, errors, typeof, numpy_support, cgutils
from numba.compiler import compile_isolated
from .support import TestCase, captured_stdout
from numba import SmartArray

def len_usecase(x):
    return len(x)

def print_usecase(x):
    print(x)

def getitem_usecase(x, key):
    return x[key]

def shape_usecase(x):
    return x.shape

def npyufunc_usecase(x):
    return np.cos(np.sin(x))

def identity(x): return x

class TestJIT(TestCase):

    def test_identity(self):
        # make sure unboxing and boxing works.
        a = SmartArray(np.arange(3))
        cfunc = jit(nopython=True)(identity)
        self.assertIs(cfunc(a),a)

    def test_len(self):
        a = SmartArray(np.arange(3))
        cfunc = jit(nopython=True)(len_usecase)
        self.assertPreciseEqual(cfunc(a), 3)

    def test_shape(self):
        a = SmartArray(np.arange(3))
        cfunc = jit(nopython=True)(shape_usecase)
        self.assertPreciseEqual(cfunc(a), (3,))

    def test_getitem(self):
        a = SmartArray(np.int32([42, 8, -5]))
        cfunc = jit(nopython=True)(getitem_usecase)
        self.assertPreciseEqual(cfunc(a, 1), 8)
        aa = cfunc(a, slice(1, None))
        self.assertIsInstance(aa, SmartArray)
        self.assertEqual(list(aa), [8, -5])

    def test_ufunc(self):
        a = SmartArray(np.int32([42, 8, -5]))
        cfunc = jit(nopython=True)(npyufunc_usecase)
        aa = cfunc(a)
        self.assertIsInstance(aa, SmartArray)
        self.assertPreciseEqual(aa.get('host'), np.cos(np.sin(a.get('host'))))

    def test_astype(self):
        a = SmartArray(np.int32([42, 8, -5]))
        aa = a.astype(np.float64)
        self.assertIsInstance(aa, SmartArray)
        # verify that SmartArray.astype() operates like ndarray.astype()...
        self.assertPreciseEqual(aa.get('host'), a.get('host').astype(np.float64))
        # ...and that both actually yield the expected dtype.
        self.assertPreciseEqual(aa.get('host').dtype.type, np.float64)
        self.assertIs(aa.dtype.type, np.float64)

class TestInterface(TestCase):

    def test_interface(self):
        # show that the SmartArray type supports all ndarray operations transparently

        a = np.arange(16).reshape(4,4)
        ap = SmartArray(a)
        ap[:,:] = 1
        ref = SmartArray(np.ones(dtype=ap.dtype, shape=(4,4)))
        eq = ap == ref
        self.assertIsInstance(eq, SmartArray)
        self.assertTrue(eq.all())

if __name__ == '__main__':
    unittest.main()
