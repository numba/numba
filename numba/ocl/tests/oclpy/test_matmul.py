from __future__ import print_function, division, absolute_import

import numpy as np

from numba import ocl, config, float32
from numba.ocl.testing import unittest

# Ensure the test takes a reasonable amount of time in the simulator
bpg, tpb = 50, 32
n = bpg * tpb
SM_SIZE = (tpb, tpb)


class TestOclMatMul(unittest.TestCase):

    def test_func(self):

        @ocl.jit(argtypes=[float32[:, ::1], float32[:, ::1], float32[:, ::1]])
        def cu_square_matrix_mul(A, B, C):
            sA = ocl.shared.array(shape=SM_SIZE, dtype=float32)
            sB = ocl.shared.array(shape=(tpb, tpb), dtype=float32)

            tx = ocl.threadIdx.x
            ty = ocl.threadIdx.y
            bx = ocl.blockIdx.x
            by = ocl.blockIdx.y
            bw = ocl.blockDim.x
            bh = ocl.blockDim.y

            x = tx + bx * bw
            y = ty + by * bh

            acc = float32(0)  # forces all the math to be f32
            for i in range(bpg):
                if x < n and y < n:
                    sA[ty, tx] = A[y, tx + i * tpb]
                    sB[ty, tx] = B[ty + i * tpb, x]

                ocl.syncthreads()

                if x < n and y < n:
                    for j in range(tpb):
                        acc += sA[ty, j] * sB[j, tx]

                ocl.syncthreads()

            if x < n and y < n:
                C[y, x] = acc

        np.random.seed(42)
        A = np.array(np.random.random((n, n)), dtype=np.float32)
        B = np.array(np.random.random((n, n)), dtype=np.float32)
        C = np.empty_like(A)

        stream = ocl.stream()
        with stream.auto_synchronize():
            dA = ocl.to_device(A, stream)
            dB = ocl.to_device(B, stream)
            dC = ocl.to_device(C, stream)
            cu_square_matrix_mul[(bpg, bpg), (tpb, tpb), stream](dA, dB, dC)
            dC.copy_to_host(C, stream)

        # Host compute
        Cans = np.dot(A, B)

        # Check result
        np.testing.assert_allclose(C, Cans, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
