from __future__ import print_function, absolute_import, division
from numba import unittest_support as unittest

import numpy as np

from numba import njit
from numba.npyufunc import dufunc
from ..support import MemoryLeakMixin


def pyuadd(a0, a1):
    return a0 + a1


class TestDUFunc(MemoryLeakMixin, unittest.TestCase):

    def nopython_dufunc(self, pyfunc):
        return dufunc.DUFunc(pyfunc, targetoptions=dict(nopython=True))

    def test_frozen(self):
        duadd = self.nopython_dufunc(pyuadd)
        self.assertFalse(duadd._frozen)
        duadd._frozen = True
        self.assertTrue(duadd._frozen)
        with self.assertRaises(ValueError):
            duadd._frozen = False
        with self.assertRaises(TypeError):
            duadd(np.linspace(0,1,10), np.linspace(1,2,10))

    def test_scalar(self):
        duadd = self.nopython_dufunc(pyuadd)
        self.assertEqual(pyuadd(1,2), duadd(1,2))

    def test_npm_call(self):
        duadd = self.nopython_dufunc(pyuadd)
        @njit
        def npmadd(a0, a1, o0):
            duadd(a0, a1, o0)
        X = np.linspace(0,1.9,20)
        X0 = X[:10]
        X1 = X[10:]
        out0 = np.zeros(10)
        npmadd(X0, X1, out0)
        np.testing.assert_array_equal(X0 + X1, out0)
        Y0 = X0.reshape((2,5))
        Y1 = X1.reshape((2,5))
        out1 = np.zeros((2,5))
        npmadd(Y0, Y1, out1)
        np.testing.assert_array_equal(Y0 + Y1, out1)
        Y2 = X1[:5]
        out2 = np.zeros((2,5))
        npmadd(Y0, Y2, out2)
        np.testing.assert_array_equal(Y0 + Y2, out2)

    def test_npm_call_implicit_output(self):
        duadd = self.nopython_dufunc(pyuadd)
        @njit
        def npmadd(a0, a1):
            return duadd(a0, a1)
        X = np.linspace(0,1.9,20)
        X0 = X[:10]
        X1 = X[10:]
        out0 = npmadd(X0, X1)
        np.testing.assert_array_equal(X0 + X1, out0)
        Y0 = X0.reshape((2,5))
        Y1 = X1.reshape((2,5))
        out1 = npmadd(Y0, Y1)
        np.testing.assert_array_equal(Y0 + Y1, out1)
        Y2 = X1[:5]
        out2 = npmadd(Y0, Y2)
        np.testing.assert_array_equal(Y0 + Y2, out2)
        out3 = npmadd(1.,2.)
        self.assertEqual(out3, 3.)

    def test_ufunc_props(self):
        duadd = self.nopython_dufunc(pyuadd)
        self.assertEqual(duadd.nin, 2)
        self.assertEqual(duadd.nout, 1)
        self.assertEqual(duadd.nargs, duadd.nin + duadd.nout)
        self.assertEqual(duadd.ntypes, 0)
        self.assertEqual(duadd.types, [])
        self.assertEqual(duadd.identity, None)
        duadd(1, 2)
        self.assertEqual(duadd.ntypes, 1)
        self.assertEqual(duadd.ntypes, len(duadd.types))

if __name__ == "__main__":
    unittest.main()
