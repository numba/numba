from .support import addtest, main
from numbapro import vectorize, float64, float32
import math
import numpy as np
import unittest

pi = math.pi

def sinc(x):
    if x == 0.0:
        return 1.0
    else:
        return math.sin(x*pi)/(pi*x)

@addtest
class TestVectorizeDecor(unittest.TestCase):
    def _run_and_compare(self, numba_func, numpy_func):
        A = np.arange(100, dtype=np.float64)
        result = numba_func(A)
        gold = numpy_func(A)
        self.assertTrue(np.allclose(result, gold))

    def test_all_cpu_based_targets(self):
        for target in ['cpu', 'stream', 'parallel']:
            numba_sinc = vectorize(['float64(float64)', 'float32(float32)'], target=target)(sinc)
            numpy_sinc = np.vectorize(sinc)
            self._run_and_compare(numba_sinc, numpy_sinc)

    def test_all_cpu_based_targets_2(self):
        for target in ['cpu', 'stream', 'parallel']:
            numba_sinc = vectorize([float64(float64), float32(float32)], target=target)(sinc)
            numpy_sinc = np.vectorize(sinc)
            self._run_and_compare(numba_sinc, numpy_sinc)

if __name__ == '__main__':
    main()