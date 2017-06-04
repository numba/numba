from __future__ import print_function, absolute_import
import numpy as np
from numba import ocl, float32
from numba.ocl.testing import unittest


def generate_input(n):
    A = np.array(np.arange(n * n).reshape(n, n), dtype=np.float32)
    B = np.array(np.arange(n) + 0, dtype=A.dtype)
    return A, B


class TestOclNonDet(unittest.TestCase):
    def test_for_pre(self):
        """Test issue with loop not running due to bad sign-extension at the for loop
        precondition.
        """

        @ocl.jit(argtypes=[float32[:, :], float32[:, :], float32[:]])
        def diagproduct(c, a, b):
            startX, startY = ocl.grid(2)
            gridX = ocl.gridDim.x * ocl.blockDim.x
            gridY = ocl.gridDim.y * ocl.blockDim.y
            height = c.shape[0]
            width = c.shape[1]

            for x in range(startX, width, (gridX)):
                for y in range(startY, height, (gridY)):
                    c[y, x] = a[y, x] * b[x]

        N = 8

        A, B = generate_input(N)

        E = np.zeros(A.shape, dtype=A.dtype)
        F = np.empty(A.shape, dtype=A.dtype)

        E = np.dot(A, np.diag(B))

        blockdim = (32, 8)
        griddim = (1, 1)

        dA = ocl.to_device(A)
        dB = ocl.to_device(B)
        dF = ocl.to_device(F, copy=False)
        diagproduct[griddim, blockdim](dF, dA, dB)
        dF.to_host()

        self.assertTrue(np.allclose(F, E))


if __name__ == '__main__':
    unittest.main()

