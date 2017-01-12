from __future__ import print_function, division, absolute_import

from timeit import default_timer as timer
import numpy as np

from numba import unittest_support as unittest
from numba import hsa, float32


class TestMatMul(unittest.TestCase):
    def test_matmul_naive(self):
        @hsa.jit
        def matmul(A, B, C):
            i = hsa.get_global_id(0)
            j = hsa.get_global_id(1)

            if i >= C.shape[0] or j >= C.shape[1]:
                return

            tmp = 0

            for k in range(A.shape[1]):
                tmp += A[i, k] * B[k, j]

            C[i, j] = tmp

        N = 256
        A = np.random.random((N, N)).astype(np.float32)
        B = np.random.random((N, N)).astype(np.float32)
        C = np.zeros_like(A)

        with hsa.register(A, B, C):
            ts = timer()
            matmul[(N // 16, N // 16), (16, 16)](A, B, C)
            te = timer()
            print("1st GPU time:", te - ts)

        with hsa.register(A, B, C):
            ts = timer()
            matmul[(N // 16, N // 16), (16, 16)](A, B, C)
            te = timer()
            print("2nd GPU time:", te - ts)

        ts = timer()
        ans = np.dot(A, B)
        te = timer()
        print("CPU time:", te - ts)
        np.testing.assert_allclose(ans, C, rtol=1e-5)

    def test_matmul_fast(self):
        blocksize = 20
        gridsize = 20

        @hsa.jit
        def matmulfast(A, B, C):
            x = hsa.get_global_id(0)
            y = hsa.get_global_id(1)

            tx = hsa.get_local_id(0)
            ty = hsa.get_local_id(1)

            sA = hsa.shared.array(shape=(blocksize, blocksize), dtype=float32)
            sB = hsa.shared.array(shape=(blocksize, blocksize), dtype=float32)

            if x >= C.shape[0] or y >= C.shape[1]:
                return

            tmp = 0

            for i in range(gridsize):
                # preload
                sA[tx, ty] = A[x, ty + i * blocksize]
                sB[tx, ty] = B[tx + i * blocksize, y]
                # wait for preload to end
                hsa.barrier(hsa.CLK_GLOBAL_MEM_FENCE)
                # compute loop
                for j in range(blocksize):
                    tmp += sA[tx, j] * sB[j, ty]
                # wait for compute to end
                hsa.barrier(hsa.CLK_GLOBAL_MEM_FENCE)

            C[x, y] = tmp

        N = gridsize * blocksize
        A = np.random.random((N, N)).astype(np.float32)
        B = np.random.random((N, N)).astype(np.float32)
        C = np.zeros_like(A)

        griddim = gridsize, gridsize
        blockdim = blocksize, blocksize

        with hsa.register(A, B, C):
            ts = timer()
            matmulfast[griddim, blockdim](A, B, C)
            te = timer()
            print("1st GPU time:", te - ts)

        with hsa.register(A, B, C):
            ts = timer()
            matmulfast[griddim, blockdim](A, B, C)
            te = timer()
            print("2nd GPU time:", te - ts)

        ts = timer()
        ans = np.dot(A, B)
        te = timer()
        print("CPU time:", te - ts)
        np.testing.assert_allclose(ans, C, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
