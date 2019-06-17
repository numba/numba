from __future__ import absolute_import, print_function, division

import math

import numpy as np

from numba import unittest_support as unittest
from numba import int32, uint32, float32, float64, jit, vectorize
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


class BaseVectorizeDecor(object):
    target = None
    wrapper = None
    funcs = {
        'func1': sinc,
        'func2': scaled_sinc,
        'func3': vector_add,
    }

    @classmethod
    def _run_and_compare(cls, func, sig, A, *args, **kwargs):
        if cls.wrapper is not None:
            func = cls.wrapper(func)
        numba_func = vectorize(sig, target=cls.target)(func)
        numpy_func = np.vectorize(func)
        result = numba_func(A, *args)
        gold = numpy_func(A, *args)
        np.testing.assert_allclose(result, gold, **kwargs)

    @tag('important')
    def test_1(self):
        sig = ['float64(float64)', 'float32(float32)']
        func = self.funcs['func1']
        A = np.arange(100, dtype=np.float64)
        self._run_and_compare(func, sig, A)

    @tag('important')
    def test_2(self):
        sig = [float64(float64), float32(float32)]
        func = self.funcs['func1']
        A = np.arange(100, dtype=np.float64)
        self._run_and_compare(func, sig, A)
    
    @tag('important')
    def test_3(self):
        sig = ['float64(float64, uint32)']
        func = self.funcs['func2']
        A = np.arange(100, dtype=np.float64)
        scale = np.uint32(3)
        self._run_and_compare(func, sig, A, scale, atol=1e-8)

    @tag('important')
    def test_4(self):
        sig = [
            int32(int32, int32),
            uint32(uint32, uint32),
            float32(float32, float32),
            float64(float64, float64),
        ]
        func = self.funcs['func3']
        A = np.arange(100, dtype=np.float64)
        self._run_and_compare(func, sig, A, A)
        A = A.astype(np.float32)
        self._run_and_compare(func, sig, A, A)
        A = A.astype(np.int32)
        self._run_and_compare(func, sig, A, A)
        A = A.astype(np.uint32)
        self._run_and_compare(func, sig, A, A)


class TestCPUVectorizeDecor(unittest.TestCase, BaseVectorizeDecor):
    target = 'cpu'


class TestParallelVectorizeDecor(unittest.TestCase, BaseVectorizeDecor):
    _numba_parallel_test_ = False
    target = 'parallel'


class TestCPUVectorizeJitted(unittest.TestCase, BaseVectorizeDecor):
    target = 'cpu'
    wrapper = staticmethod(jit)  # staticmethod required for py27


if __name__ == '__main__':
    unittest.main()
