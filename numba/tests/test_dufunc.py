from __future__ import print_function, absolute_import, division
from numba import unittest_support as unittest

import numpy as np

from numba import njit
from numba.npyufunc import dufunc

def pyuadd(a0, a1):
    return a0 + a1

class TestDUFunc(unittest.TestCase):

    def test_frozen(self):
        duadd = dufunc.DUFunc(pyuadd, nopython=True)
        self.assertFalse(duadd.frozen)
        duadd.frozen = True
        self.assertTrue(duadd.frozen)
        with self.assertRaises(ValueError):
            duadd.frozen = False
        with self.assertRaises(TypeError):
            duadd(np.linspace(0,1,10), np.linspace(1,2,10))

    def test_scalar(self):
        duadd = dufunc.DUFunc(pyuadd, nopython=True)
        self.assertEqual(pyuadd(1,2), duadd(1,2))

    def test_npm_call(self):
        duadd = dufunc.DUFunc(pyuadd, nopython=True)
        @njit
        def npmadd(a0, a1, o0):
            duadd(a0, a1, o0)
        X = np.linspace(0,1.9,20)
        X0 = X[:10]
        X1 = X[10:]
        out0 = np.zeros(10)
        npmadd(X0, X1, out0)
        self.assertTrue(np.all(X0 + X1 == out0))
        Y0 = X0.reshape((2,5))
        Y1 = X1.reshape((2,5))
        out1 = np.zeros((2,5))
        npmadd(Y0, Y1, out1)
        self.assertTrue(np.all(Y0 + Y1 == out1))
        Y2 = X1[:5]
        out2 = np.zeros((2,5))
        npmadd(Y0, Y2, out2)
        self.assertTrue(np.all(Y0 + Y2 == out2))

if __name__ == "__main__":
    unittest.main()
