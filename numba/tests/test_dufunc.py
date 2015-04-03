from __future__ import print_function, absolute_import, division
from numba import unittest_support as unittest

import numpy as np

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

if __name__ == "__main__":
    unittest.main()
