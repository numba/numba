import math
import numpy as np
import unittest
#import logging; logging.getLogger().setLevel(1)

from numba import *

def test_exp(a):
    return math.exp(a)

def test_sqrt(a):
    return math.sqrt(a)

def test_log(a):
    return math.log(a)

class TestNoPythonMath(unittest.TestCase):
    def test_sqrt(self):
        self._template(test_sqrt, np.sqrt)

    def test_exp(self):
        self._template(test_exp, np.exp)

    def test_log(self):
        self._template(test_log, np.log)

    def _template(self, func, npfunc):
        func_jitted = jit(argtypes=[f4], restype=f4, nopython=True)(func)
        A = np.array(np.random.random(10), dtype=np.float32)
        B = np.vectorize(func_jitted)(A)
        self.assertTrue(np.allclose(B, npfunc(A)))


if __name__ == '__main__':
    unittest.main()