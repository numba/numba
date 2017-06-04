from __future__ import absolute_import, print_function, division
from numba import vectorize
from numba import ocl, float32
import numpy as np
from numba import unittest_support as unittest
from numba.ocl.testing import skip_on_oclsim


class TestOclVectorizeDeviceCall(unittest.TestCase):
    def test_ocl_vectorize_device_call(self):

        @ocl.jit(float32(float32, float32, float32), device=True)
        def cu_device_fn(x, y, z):
            return x ** y / z

        def cu_ufunc(x, y, z):
            return cu_device_fn(x, y, z)

        ufunc = vectorize([float32(float32, float32, float32)], target='ocl')(
            cu_ufunc)

        N = 100

        X = np.array(np.random.sample(N), dtype=np.float32)
        Y = np.array(np.random.sample(N), dtype=np.float32)
        Z = np.array(np.random.sample(N), dtype=np.float32) + 0.1

        out = ufunc(X, Y, Z)

        gold = (X ** Y) / Z

        self.assertTrue(np.allclose(out, gold))


if __name__ == '__main__':
    unittest.main()
