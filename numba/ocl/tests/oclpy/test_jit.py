from __future__ import print_function, absolute_import
from numba import unittest_support as unittest
from numba import ocl
import numpy as np


@ocl.jit("int32(int32, int32)", device=True)
def ocl_add(a, b):
    return a + b


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

        self.assertTrue(np.all(data == np.arange(n)))

    def test_device_fn(self):
        @ocl.jit("void(int32[::1], int32[::1], int32[::1])")
        def ocl_add_kernel(x, y, z):
            i = ocl.get_global_id(0)
            z[i] = ocl_add(x[i], y[i])

        n = 10
        x = np.arange(n, dtype='int32')
        y = np.arange(x.size, dtype='int32')
        z = np.zeros_like(x)

        dev_x = ocl.to_device(x)
        dev_y = ocl.to_device(y)
        dev_z = ocl.to_device(z)

        ocl_add_kernel.configure(n)(dev_x, dev_y, dev_z)
        dev_z.copy_to_host(z)

        print(z)
        self.assertTrue(np.all(z == x + y))


if __name__ == '__main__':
    unittest.main()
