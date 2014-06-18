from __future__ import print_function
from numba import unittest_support as unittest
from numba import cuda
import numpy as np


class TestCudaAutojit(unittest.TestCase):
    def test_device_array(self):
        @cuda.autojit
        def foo(x, y):
            i = cuda.grid(1)
            y[i] = x[i]

        x = np.arange(10)
        y = np.empty_like(x)

        dx = cuda.to_device(x)
        dy = cuda.to_device(y)

        foo[10, 1](dx, dy)

        dy.copy_to_host(y)

        self.assertTrue(np.all(x == y))


if __name__ == '__main__':
    unittest.main()
