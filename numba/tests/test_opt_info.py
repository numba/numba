import math
import numpy as np
import unittest

from numba import njit
from numba.tests.support import TestCase
from numba.core.opt_info import RawOptimizationRemarks, LoopDeLoop,\
    SuperWorldLevelParallelismDetector


class TestOptimizationInfo(TestCase):

    def test_usage(self):
        @njit(opt_info=[RawOptimizationRemarks()])
        def foo(n):
            c = 0
            for i in range(n):
                c += i
            return c

        foo(3)
        self.assertTrue(all('raw' in metadata['opt_info'] for metadata in
                            foo.get_metadata().values()))
        # Generate a new signature and make sure that everything still has info
        foo(3.0)
        self.assertTrue(all('raw' in metadata['opt_info'] for metadata in
                            foo.get_metadata().values()))

    def test_loop_vect(self):
        @njit(opt_info=[LoopDeLoop()])
        def foo(n):
            ret = np.empty(n, dtype=np.float64)
            for x in range(n):
                ret[x] = math.sin(np.float64(x))
            return ret

        foo(3)
        # Check that there is data for the loop
        self.assertTrue(all(metadata['opt_info']['loop_vectorization']
                            for metadata in foo.get_metadata().values()))
        # Check that all the loops got vectorized
        self.assertTrue(all(v for metadata in foo.get_metadata().values() for v
                            in metadata['opt_info']['loop_vectorization']
                            .values()))

    def test_slp(self):
        # Sample translated from:
        # https://www.llvm.org/docs/Vectorizers.html#the-slp-vectorizer

        @njit(opt_info=[SuperWorldLevelParallelismDetector()])
        def foo(a1, a2, b1, b2):
            A = np.empty(4)
            A[0] = a1 * (a1 + b1)
            A[1] = a2 * (a2 + b2)
            A[2] = a1 * (a1 + b1)
            A[3] = a2 * (a2 + b2)
            return A

        foo(3.0, 4.0, 5.0, 6.0)
        self.assertTrue(all(metadata['opt_info']['slp_vectorization'] for
                            metadata in foo.get_metadata().values()))


if __name__ == "__main__":
    unittest.main()
