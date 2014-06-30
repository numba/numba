from __future__ import print_function, absolute_import
from numba import unittest_support as unittest
from numba import vectorize, ocl
import numpy as np


class TestVectorize(unittest.TestCase):
    def test_ufunc_builder(self):
        def sum(a, b):
            return a + b

        vsum = vectorize(["int32(int32, int32)"], target='ocl')(sum)

        n = 10
        a = np.arange(n, dtype="int32")
        b = np.arange(a.size, dtype="int32")
        c = np.zeros_like(a)

        dev_a = ocl.to_device(a)
        dev_b = ocl.to_device(b)
        dev_c = ocl.to_device(c)
        vsum(dev_a, dev_b, out=dev_c)
        dev_c.copy_to_host(c)

        print(c)
        self.assertTrue(np.all(a + b == c))


if __name__ == '__main__':
    unittest.main()

