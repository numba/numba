from __future__ import print_function, division, absolute_import

import numpy as np

from numba.cuda.testing import unittest, SerialMixin
from numba import cuda


class TestCudaArray(SerialMixin, unittest.TestCase):
    def test_gpu_array_zero_length(self):
        x = np.arange(0)
        dx = cuda.to_device(x)
        hx = dx.copy_to_host()
        self.assertEqual(x.shape, dx.shape)
        self.assertEqual(x.size, dx.size)
        self.assertEqual(x.shape, hx.shape)
        self.assertEqual(x.size, hx.size)

    def test_gpu_array_strided(self):

        @cuda.jit('void(double[:])')
        def kernel(x):
            i = cuda.grid(1)
            if i < x.shape[0]:
                x[i] = i

        x = np.arange(10, dtype=np.double)
        y = np.ndarray(shape=10 * 8, buffer=x, dtype=np.byte)
        z = np.ndarray(9, buffer=y[4:-4], dtype=np.double)
        kernel[10, 10](z)
        self.assertTrue(np.allclose(z, list(range(9))))

    def test_gpu_array_interleaved(self):

        @cuda.jit('void(double[:], double[:])')
        def copykernel(x, y):
            i = cuda.grid(1)
            if i < x.shape[0]:
                x[i] = i
                y[i] = i

        x = np.arange(10, dtype=np.double)
        y = x[:-1:2]
        # z = x[1::2]
        # n = y.size
        try:
            cuda.devicearray.auto_device(y)
        except ValueError:
            pass
        else:
            raise AssertionError("Should raise exception complaining the "
                                 "contiguous-ness of the array.")
            # Should we handle this use case?
            # assert z.size == y.size
            # copykernel[1, n](y, x)
            # print(y, z)
            # assert np.all(y == z)
            # assert np.all(y == list(range(n)))

    def test_auto_device_const(self):
        d, _ = cuda.devicearray.auto_device(2)
        self.assertTrue(np.all(d.copy_to_host() == np.array(2)))

if __name__ == '__main__':
    unittest.main()
