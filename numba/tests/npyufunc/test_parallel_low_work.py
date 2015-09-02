"""
There was a deadlock problem when work count is smaller than number of threads.
"""
from __future__ import absolute_import, print_function, division
from numba import unittest_support as unittest
import numpy as np
from numba import float32, float64, int32, uint32
from numba.npyufunc import Vectorize
from timeit import default_timer as time


def vector_add(a, b):
    return a + b


class TestParallelLowWorkCount(unittest.TestCase):
    def test_low_workcount(self):
        # build parallel native code ufunc
        pv = Vectorize(vector_add, target='parallel')
        pv.add(restype=int32, argtypes=[int32, int32])
        pv.add(restype=uint32, argtypes=[uint32, uint32])
        pv.add(restype=float32, argtypes=[float32, float32])
        pv.add(restype=float64, argtypes=[float64, float64])
        para_ufunc = pv.build_ufunc()

        # build python ufunc
        np_ufunc = np.vectorize(vector_add)

        # test it out
        def test(ty):
            data = np.arange(1).astype(ty) # just one item
            result = para_ufunc(data, data)
            gold = np_ufunc(data, data)
            self.assertTrue(np.allclose(gold, result))

        test(np.double)
        test(np.float32)
        test(np.int32)
        test(np.uint32)


if __name__ == '__main__':
    unittest.main()
