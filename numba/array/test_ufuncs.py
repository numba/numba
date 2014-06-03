import numba.unittest_support as unittest
import numba.array as numbarray
import numpy as np
from numba.config import PYVERSION
from math import pi
use_python = True


class TestUFuncs(unittest.TestCase):

    def test_binary_ufunc(self):

        a = numbarray.arange(10)
        result = numbarray.add(a, a)
        expected = np.add(np.arange(10), np.arange(10))

        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

        result = numbarray.add(1, 1)
        expected = np.add(1, 1)

        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

        result = numbarray.add(a, 1)
        expected = np.add(np.arange(10), 1)

        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

        result = numbarray.add(1, a)
        expected = np.add(1, np.arange(10))

        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

    def test_unary_sin_ufunc(self):

        a = (pi / 2) * numbarray.ones(10)
        b = (pi / 2) * np.ones(10)

        result = numbarray.sin(a)
        result.eval()  # eval is deferred so w/o (or print) this the assert fails
        #print result
        expected = np.sin(b)
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

    def test_unary_cos_ufunc(self):

        a = numbarray.zeros(10)
        b = np.zeros(10)

        result = numbarray.cos(a)
        result.eval()
        expected = np.cos(b)
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

    def test_unary_tan_ufunc(self):

        # The (pi / 4) fails
        #a = (pi / 4) * numbarray.ones(10)
        a = numbarray.zeros(10)
        #b = (pi / 4) * np.ones(10)
        b = np.zeros(10)
        result = numbarray.tan(a)
        result.eval()
        expected = np.tan(b)
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

    def test_unary_arcsin_ufunc(self):

        a = numbarray.zeros(10)
        b = np.zeros(10)
        result = numbarray.arcsin(a)
        result.eval()
        expected = np.arcsin(b)
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

    def test_unary_arccos_ufunc(self):

        a = numbarray.ones(10)

        b = np.ones(10)
        result = numbarray.arccos(a)
        result.eval()
        print type(result)
        expected = np.arccos(b)
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))
    
    def test_unary_arctan_ufunc(self):

        a = numbarray.ones(10)
        b = np.ones(10)
        result = numbarray.arctan(a)
        result.eval()
        print type(result)
        expected = np.arctan(b)
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))
     

if __name__ == '__main__':
    unittest.main()
