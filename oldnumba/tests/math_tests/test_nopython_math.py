import sys

import math
import numpy as np
import unittest
#import logging; logging.getLogger().setLevel(1)

from numba import *

def exp_fn(a):
    return math.exp(a)

def sqrt_fn(a):
    return math.sqrt(a)

def log_fn(a):
    return math.log(a)

class TestNoPythonMath(unittest.TestCase):
    def test_sqrt(self):
        self._template(sqrt_fn, np.sqrt)

    def test_exp(self):
        self._template(exp_fn, np.exp)

    def test_log(self):
        self._template(log_fn, np.log)

    def _template(self, func, npfunc):
        func_jitted = jit(argtypes=[f4], restype=f4, nopython=True)(func)
        A = np.array(np.random.random(10), dtype=np.float32)
        B = np.vectorize(func_jitted)(A)
        self.assertTrue(np.allclose(B, npfunc(A)))

if sys.platform == 'win32':
    # NOTE: we're using the double implementation (e.g. 'log' instead of 'logf')
    # class TestNoPythonMath(unittest.TestCase):
    #     """
    #     LLVM intrinsics don't work properly on Windows, and libc doesn't
    #     have all these functions.
    #     """
    pass

if __name__ == '__main__':
    unittest.main()
