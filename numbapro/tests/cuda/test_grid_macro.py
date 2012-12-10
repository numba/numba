import unittest
import numpy as np
from numba import *
from numbapro import cuda
import support

@jit(argtypes=[f4[:]], target='gpu')
def cu_array_double_1d(dst):
    i = cuda.grid(1)
    dst[i] = dst[i] * 2

@jit(argtypes=[f4[:, :]], target='gpu')
def cu_array_double_2d(dst):
    i, j = cuda.grid(2)
    dst[i, j] = dst[i, j] * 2

def cu_array_double_3d(dst):
    i, j, k = cuda.grid(3)   # raises error
    dst[i, j, k] = dst[i, j, k] * 2


class TestCudaMacro(support.CudaTestCase):
    def test_grid_1d(self):
        A = np.array(np.random.random(256), dtype=np.float32)
        Gold = A * 2

        cu_array_double_1d[(1,), (A.shape[0],)](A)

        for i, (got, expect) in enumerate(zip(A, Gold)):
            self.assertEqual(got, expect, "%s != %s at i=%d" % (got, expect, i))

    def test_grid_2d(self):
        A = np.array(np.random.random(16**2), dtype=np.float32).reshape(16, 16)
        Gold = A * 2

        cu_array_double_2d[(1,), A.shape](A)

        for i, (got, expect) in enumerate(zip(A.flatten(), Gold.flatten())):
            self.assertEqual(got, expect, "%s != %s at i=%d" % (got, expect, i))

    def test_grid_3d(self):
        # this just complains
        with self.assertRaises(ValueError):
            jit(argtypes=[f4[:, :]], target='gpu')(cu_array_double_3d)

                      
if __name__ == '__main__':
    unittest.main()


