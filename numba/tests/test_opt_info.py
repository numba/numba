import math
import numpy as np
import unittest

from numba import njit
from numba.tests.support import TestCase
from numba.core.opt_info import RawOptimizationRemarks, LoopDeLoop


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


if __name__ == "__main__":
    unittest.main()
