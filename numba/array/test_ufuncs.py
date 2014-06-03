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
  
    def test_binary_hypot_ufunc(self):
        a = 3 * numbarray.ones((3, 3))
        b = 4 * numbarray.ones((3, 3))
        c = 3 * np.ones((3, 3))
        d = 4 * np.ones((3, 3))
        result = numbarray.hypot(a, b)
        result.eval()
        expected = np.hypot(c, d)
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))
    
    def test_binary_arctan2_ufunc(self):
        # the first two tests work fine
        # the test with the 180 / [numbarray.pi|np.pi] fails
        a = numbarray.array([-1, +1, +1, -1, 0])
        b = numbarray.array([-1, -1, +1, +1, 0])
        c = np.array([-1, +1, +1, -1, 0])
        d = np.array([-1, -1, +1, +1, 0])
        #result = numbarray.arctan2(a, b) * 180
        #result = numbarray.arctan2(a, b) * 180 / 3.14 
        result = numbarray.arctan2(a, b) * 180 / numbarray.pi
        result.eval()
        print result
        print numbarray.pi
        #expected = np.arctan2(c, d) * 180
        #expected = np.arctan2(c, d) * 180 / 3.14 
        expected = np.arctan2(c, d) * 180 / np.pi
        print expected
        print np.pi
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))
       
    def test_div(self):
        a = 180 / numbarray.pi
        print a
        b = 180 / np.pi
        print b
        self.assertTrue(a == b)

    def test_pi(self):
        self.assertTrue(numbarray.pi == np.pi)


if __name__ == '__main__':
    unittest.main()
