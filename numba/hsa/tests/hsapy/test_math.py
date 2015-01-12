from __future__ import print_function, absolute_import

import numpy as np
import math

import numba.unittest_support as unittest
from numba import hsa


class TestMath(unittest.TestCase):
    def test_sin(self):
        @hsa.jit
        def sin(dst, src):
            i = hsa.get_global_id(0)
            if i < dst.size:
                dst[i] = math.sin(src[i])

        these_types = [np.float32, np.float64]
        for dtype in these_types:
            src = np.arange(10, dtype=dtype)
            dst = np.zeros_like(src)

            sin[src.size, 1](dst, src)

            np.testing.assert_allclose(dst, np.sin(src))

    def test_cos(self):
        @hsa.jit
        def cos(dst, src):
            i = hsa.get_global_id(0)
            if i < dst.size:
                dst[i] = math.cos(src[i])

        these_types = [np.float32, np.float64]
        for dtype in these_types:
            src = np.arange(10, dtype=dtype)
            dst = np.zeros_like(src)

            cos[src.size, 1](dst, src)

            np.testing.assert_allclose(dst, np.cos(src), rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
