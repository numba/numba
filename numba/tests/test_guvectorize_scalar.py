"""
Tests for guvectorize scalar arguments
"""
from __future__ import print_function, absolute_import, division
import numpy as np
from numba import guvectorize
from numba import unittest_support as unittest


class TestGUVectorizeScalar(unittest.TestCase):
    """
    Nothing keeps user from out-of-bound memory access
    """

    def test_scalar_output(self):
        """
        Note that scalar output is a 0-dimension array that acts as
        a pointer to the output location.
        """

        @guvectorize(['void(int32[:], int32[:])'], '(n)->()')
        def sum_row(inp, out):
            tmp = 0.
            for i in range(inp.shape[0]):
                tmp += inp[i]
            out[()] = tmp

        # inp is (10000, 3)
        # out is (10000)
        # The outter (leftmost) dimension must match or numpy broadcasting is performed.

        inp = np.arange(30000, dtype=np.int32).reshape(10000, 3)
        out = sum_row(inp)

        # verify result
        for i in range(inp.shape[0]):
            assert out[i] == inp[i].sum()

    def test_scalar_input(self):

        @guvectorize(['int32[:], int32[:], int32[:]'], '(n),()->(n)')
        def foo(inp, n, out):
            for i in range(inp.shape[0]):
                out[i] = inp[i] * n[()]

        inp = np.arange(3 * 10, dtype=np.int32).reshape(10, 3)
        # out = np.empty_like(inp)
        out = foo(inp, 2)

        # verify result
        self.assertTrue(np.all(inp * 2 == out))

if __name__ == '__main__':
    unittest.main()

