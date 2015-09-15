from __future__ import absolute_import, print_function, division
from numba import unittest_support as unittest
import numpy as np
from numba import int32, uint32, float32, float64
from numba import vectorize
from timeit import default_timer as time


def vector_add(a, b):
    return a + b


def template_vectorize(self, target):
    # build basic native code ufunc
    sig = [int32(int32, int32),
           uint32(uint32, uint32),
           float32(float32, float32),
           float64(float64, float64)]
    basic_ufunc = vectorize(sig, target=target)(vector_add)

    # build python ufunc
    np_ufunc = np.add

    # test it out
    def test(ty):
        data = np.linspace(0., 100., 500).astype(ty)
        result = basic_ufunc(data, data)
        gold = np_ufunc(data, data)
        self.assertTrue(np.allclose(gold, result))

    test(np.double)
    test(np.float32)
    test(np.int32)
    test(np.uint32)


class TestVectorize(unittest.TestCase):
    def test_basic_vectorize(self):
        template_vectorize(self, 'cpu')

    def test_parallel_vectorize(self):
        template_vectorize(self, 'parallel')


if __name__ == '__main__':
    unittest.main()
