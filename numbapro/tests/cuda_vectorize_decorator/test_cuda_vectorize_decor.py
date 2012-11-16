from numbapro import vectorize
from numba import *
import math
import numpy as np
import unittest

def sqrtc(x):
    if x == 0.0:
        return 1.0
    else:
        return math.sqrt(x)

class TestVectorizeDecor(unittest.TestCase):
    def _run_and_compare(self, numba_func, numpy_func):
        A = np.arange(100, dtype=np.float64)
        result = numba_func(A)
        gold = numpy_func(A)
        self.assertTrue(np.allclose(result, gold))

    def test_cuda_vectorize(self):
        numba_sqrtc = vectorize([f8(f8), f4(f4)], target='gpu')(sqrtc)
        numpy_sqrtc = np.vectorize(sqrtc)
        self._run_and_compare(numba_sqrtc, numpy_sqrtc)

if __name__ == '__main__':
    unittest.main()