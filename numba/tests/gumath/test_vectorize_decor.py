from __future__ import absolute_import, print_function, division

import math

import numpy as np

from numba import unittest_support as unittest
from numba import int32, uint32, float32, float64, jit
from numba.gumath import jit_xnd
from ..support import tag
from xnd import xnd

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


class TestVectorizeDecor(unittest.TestCase):
    funcs = {
        'func1': sinc,
        'func2': scaled_sinc,
        'func3': vector_add,
    }

    @classmethod
    def _run_and_compare(cls, func, sig, *args, **kwargs):
        numba_func = jit_xnd(sig)(func)
        numpy_func = np.vectorize(func)
        result = numba_func(*args)
        gold = numpy_func(*args)
        np.testing.assert_allclose(result, gold, **kwargs)

    @tag('important')
    def test_1(self):
        sig = '... * float64 -> ... * float64'
        func = self.funcs['func1']
        A = np.arange(100, dtype=np.float64)
        self._run_and_compare(func, sig, A)

    @tag('important')
    def test_2(self):
        sig = '... * float64 -> ... * float64'
        func = self.funcs['func1']
        A = np.arange(100, dtype=np.float64)
        self._run_and_compare(func, sig, A)
    
    @tag('important')
    def test_3(self):
        sig = '... * float64, ... * uint8 -> ... * float64'
        func = self.funcs['func2']
        A = np.arange(100, dtype=np.float64)
        # should be uint32 once https://github.com/numpy/numpy/pull/10564 is released
        scale = np.uint8(3)
        self._run_and_compare(func, sig, A, scale, atol=1e-8)

    @tag('important')
    def test_4(self):
        sig = '... * D, ... * D -> ... * D'
        func = self.funcs['func3']
        A = np.arange(100, dtype=np.float64)
        self._run_and_compare(func, sig, A, A)
        A = A.astype(np.float32)
        self._run_and_compare(func, sig, A, A)
        A = A.astype(np.int32)
        self._run_and_compare(func, sig, A, A)
        A = A.astype(np.uint32)
        self._run_and_compare(func, sig, A, A)

if __name__ == '__main__':
    unittest.main()
