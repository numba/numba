from __future__ import print_function
import numpy as np
from numba import cuda
from numba import unittest_support as unittest

class TestReduction(unittest.TestCase):
    def test_sum_reduce(self):
        sum_reduce = cuda.Reduce(lambda a, b: a + b)
        A = (np.arange(34567, dtype=np.float64) + 1)
        expect = A.sum()
        got = sum_reduce(A)
        self.assertEqual(expect, got)

    def test_empty_array_host(self):
        sum_reduce = cuda.Reduce(lambda a, b: a + b)
        A = (np.arange(0, dtype=np.float64) + 1)
        expect = A.sum()
        got = sum_reduce(A)
        self.assertEqual(expect, got)

    def test_empty_array_device(self):
        sum_reduce = cuda.Reduce(lambda a, b: a + b)
        A = (np.arange(0, dtype=np.float64) + 1)
        dA = cuda.to_device(A)
        expect = A.sum()
        got = sum_reduce(dA)
        self.assertEqual(expect, got)

    def test_prod_reduce(self):
        prod_reduce = cuda.reduce(lambda a, b: a * b)
        A = (np.arange(64, dtype=np.float64) + 1)
        expect = A.prod()
        got = prod_reduce(A, init=1)
        self.assertTrue(np.allclose(expect, got))

    def test_max_reduce(self):
        max_reduce = cuda.Reduce(lambda a, b: max(a, b))
        A = (np.arange(3717, dtype=np.float64) + 1)
        expect = A.max()
        got = max_reduce(A, init=0)
        self.assertEqual(expect, got)


if __name__ == '__main__':
    unittest.main()
