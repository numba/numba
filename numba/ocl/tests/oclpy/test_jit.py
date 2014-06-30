from __future__ import print_function, absolute_import
from numba import unittest_support as unittest
from numba import ocl
import numpy as np


class TestJIT(unittest.TestCase):
    def test_array_kernel(self):
        @ocl.jit("void(int32[::1])")
        def oclkernel(x):
            i = ocl.get_global_id(0)
            x[i] = i

        n = 10
        data = np.zeros(n, dtype='int32')

        dev_data = ocl.to_device(data)
        oclkernel.configure(n)(dev_data)
        dev_data.copy_to_host(data)

        print(data)
        self.assertTrue(np.all(data == np.arange(n)))


if __name__ == '__main__':
    unittest.main()

