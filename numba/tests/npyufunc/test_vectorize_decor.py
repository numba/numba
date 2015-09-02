from __future__ import absolute_import, print_function, division
from numba import unittest_support as unittest
from numba import vectorize, float64, float32
import math
import numpy as np

pi = math.pi


def sinc(x):
    if x == 0.0:
        return 1.0
    else:
        return math.sin(x * pi) / (pi * x)

def scaled_sinc(x, scale):
    if x == 0.0:
        return scale
    else:
        return scale * (math.sin(x * pi) / (pi * x))


class BaseVectorizeDecor(unittest.TestCase):
    def _run_and_compare(self, numba_func, numpy_func):
        A = np.arange(100, dtype=np.float64)
        result = numba_func(A)
        gold = numpy_func(A)
        self.assertTrue(np.allclose(result, gold))

    def _test_template_1(self, target):
        numba_sinc = vectorize(['float64(float64)', 'float32(float32)'],
                               target=target)(sinc)
        numpy_sinc = np.vectorize(sinc)
        self._run_and_compare(numba_sinc, numpy_sinc)

    def _test_template_2(self, target):
        numba_sinc = vectorize([float64(float64), float32(float32)],
                               target=target)(sinc)
        numpy_sinc = np.vectorize(sinc)
        self._run_and_compare(numba_sinc, numpy_sinc)

    def _test_template_3(self, target):
        numba_scaled_sinc = vectorize(['float64(float64, uint32)'],
                                      target=target)(scaled_sinc)
        numpy_scaled_sinc = np.vectorize(scaled_sinc)
        A = np.arange(100, dtype=np.float64)
        scale = np.uint32(3)
        result = numba_scaled_sinc(A, scale)
        gold = numpy_scaled_sinc(A, scale)
        self.assertTrue(np.allclose(result, gold))


class TestVectorizeDecor(BaseVectorizeDecor):

    def test_cpu_1(self):
        self._test_template_1('cpu')

    def test_parallel_1(self):
        self._test_template_1('parallel')

    def test_cpu_2(self):
        self._test_template_2('cpu')

    def test_parallel_2(self):
        self._test_template_2('parallel')

    def test_cpu_3(self):
        self._test_template_3('cpu')

    def test_parallel_3(self):
        self._test_template_3('parallel')


if __name__ == '__main__':
    unittest.main()
