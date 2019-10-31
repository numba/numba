from __future__ import print_function, absolute_import

import numpy as np

from numba import cuda
import numba.unittest_support as unittest
from numba.cuda.testing import SerialMixin


class TestForAll(SerialMixin, unittest.TestCase):
    def test_forall_1(self):
        @cuda.jit
        def foo(x):
            i = cuda.grid(1)
            if i < x.size:
                x[i] += 1

        arr = np.arange(11)
        orig = arr.copy()
        foo.forall(arr.size)(arr)
        np.testing.assert_array_almost_equal(arr, orig + 1)

    def test_forall_2(self):
        @cuda.jit("void(float32, float32[:], float32[:])")
        def bar(a, x, y):
            i = cuda.grid(1)
            if i < x.size:
                y[i] = a * x[i] + y[i]

        x = np.arange(13, dtype=np.float32)
        y = np.arange(13, dtype=np.float32)
        oldy = y.copy()
        a = 1.234
        bar.forall(y.size)(a, x, y)
        np.testing.assert_array_almost_equal(y, a * x + oldy, decimal=3)


if __name__ == '__main__':
    unittest.main()
