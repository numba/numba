from __future__ import absolute_import, print_function, division

import math

import numpy as np

from numba import unittest_support as unittest
from numba import int32, uint32, float32, float64, vectorize
from ..support import tag


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

def vector_add(a, b):
    return a + b


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

    def _test_template_4(self, target):
        sig = [int32(int32, int32),
               uint32(uint32, uint32),
               float32(float32, float32),
               float64(float64, float64)]
        basic_ufunc = vectorize(sig, target=target)(vector_add)
        np_ufunc = np.add

        def test(ty):
            data = np.linspace(0., 100., 500).astype(ty)
            result = basic_ufunc(data, data)
            gold = np_ufunc(data, data)
            self.assertTrue(np.allclose(gold, result))

        test(np.double)
        test(np.float32)
        test(np.int32)
        test(np.uint32)


class TestVectorizeDecor(BaseVectorizeDecor):

    @tag('important')
    def test_cpu_1(self):
        self._test_template_1('cpu')

    @tag('important')
    def test_parallel_1(self):
        self._test_template_1('parallel')

    @tag('important')
    def test_cpu_2(self):
        self._test_template_2('cpu')

    @tag('important')
    def test_parallel_2(self):
        self._test_template_2('parallel')

    @tag('important')
    def test_cpu_3(self):
        self._test_template_3('cpu')

    @tag('important')
    def test_parallel_3(self):
        self._test_template_3('parallel')

    @tag('important')
    def test_cpu_4(self):
        self._test_template_4('cpu')

    @tag('important')
    def test_parallel_4(self):
        self._test_template_4('parallel')


if __name__ == '__main__':
    unittest.main()
