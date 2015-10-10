from __future__ import print_function, absolute_import
from numba import void, float32
import numpy as np
import numpy.core.umath_tests as ut
from numba import guvectorize
from numba import cuda
from timeit import default_timer as time
from numba import unittest_support as unittest
from numba.cuda.testing import skip_on_cudasim


non_stream_speedups = []
stream_speedups = []


@skip_on_cudasim('ufunc API unsupported in the simulator')
class TestCUDAGufunc(unittest.TestCase):

    def test_gufunc_small(self):

        @guvectorize([void(float32[:, :], float32[:, :], float32[:, :])],
                     '(m,n),(n,p)->(m,p)',
                     target='cuda')
        def matmulcore(A, B, C):
            m, n = A.shape
            n, p = B.shape
            for i in range(m):
                for j in range(p):
                    C[i, j] = 0
                    for k in range(n):
                        C[i, j] += A[i, k] * B[k, j]


        gufunc = matmulcore
        gufunc.max_blocksize = 512

        matrix_ct = 2
        A = np.arange(matrix_ct * 2 * 4, dtype=np.float32).reshape(matrix_ct, 2,
                                                                   4)
        B = np.arange(matrix_ct * 4 * 5, dtype=np.float32).reshape(matrix_ct, 4,
                                                                   5)

        ts = time()
        C = gufunc(A, B)
        tcuda = time() - ts

        ts = time()
        Gold = ut.matrix_multiply(A, B)
        tcpu = time() - ts

        non_stream_speedups.append(tcpu / tcuda)

        print(C, Gold)

        self.assertTrue(np.allclose(C, Gold))

    def test_gufunc_auto_transfer(self):

        @guvectorize([void(float32[:, :], float32[:, :], float32[:, :])],
                     '(m,n),(n,p)->(m,p)',
                     target='cuda')
        def matmulcore(A, B, C):
            m, n = A.shape
            n, p = B.shape
            for i in range(m):
                for j in range(p):
                    C[i, j] = 0
                    for k in range(n):
                        C[i, j] += A[i, k] * B[k, j]

        gufunc = matmulcore
        gufunc.max_blocksize = 512

        matrix_ct = 2
        A = np.arange(matrix_ct * 2 * 4, dtype=np.float32).reshape(matrix_ct, 2,
                                                                   4)
        B = np.arange(matrix_ct * 4 * 5, dtype=np.float32).reshape(matrix_ct, 4,
                                                                   5)

        dB = cuda.to_device(B)

        ts = time()
        C = gufunc(A, dB).copy_to_host()
        tcuda = time() - ts

        ts = time()
        Gold = ut.matrix_multiply(A, B)
        tcpu = time() - ts

        non_stream_speedups.append(tcpu / tcuda)

        print(C, Gold)

        self.assertTrue(np.allclose(C, Gold))

    def test_gufunc(self):

        @guvectorize([void(float32[:, :], float32[:, :], float32[:, :])],
                     '(m,n),(n,p)->(m,p)',
                     target='cuda')
        def matmulcore(A, B, C):
            m, n = A.shape
            n, p = B.shape
            for i in range(m):
                for j in range(p):
                    C[i, j] = 0
                    for k in range(n):
                        C[i, j] += A[i, k] * B[k, j]

        gufunc = matmulcore
        gufunc.max_blocksize = 512

        matrix_ct = 1001 # an odd number to test thread/block division in CUDA
        A = np.arange(matrix_ct * 2 * 4, dtype=np.float32).reshape(matrix_ct, 2,
                                                                   4)
        B = np.arange(matrix_ct * 4 * 5, dtype=np.float32).reshape(matrix_ct, 4,
                                                                   5)

        ts = time()
        C = gufunc(A, B)
        tcuda = time() - ts

        ts = time()
        Gold = ut.matrix_multiply(A, B)
        tcpu = time() - ts

        non_stream_speedups.append(tcpu / tcuda)

        self.assertTrue(np.allclose(C, Gold))

    def test_gufunc_hidim(self):

        @guvectorize([void(float32[:, :], float32[:, :], float32[:, :])],
                     '(m,n),(n,p)->(m,p)',
                     target='cuda')
        def matmulcore(A, B, C):
            m, n = A.shape
            n, p = B.shape
            for i in range(m):
                for j in range(p):
                    C[i, j] = 0
                    for k in range(n):
                        C[i, j] += A[i, k] * B[k, j]

        gufunc = matmulcore
        gufunc.max_blocksize = 512

        matrix_ct = 100 # an odd number to test thread/block division in CUDA
        A = np.arange(matrix_ct * 2 * 4, dtype=np.float32).reshape(4, 25, 2, 4)
        B = np.arange(matrix_ct * 4 * 5, dtype=np.float32).reshape(4, 25, 4, 5)

        ts = time()
        C = gufunc(A, B)
        tcuda = time() - ts

        ts = time()
        Gold = ut.matrix_multiply(A, B)
        tcpu = time() - ts

        non_stream_speedups.append(tcpu / tcuda)

        self.assertTrue(np.allclose(C, Gold))

    def test_gufunc_adjust_blocksize(self):

        @guvectorize([void(float32[:, :], float32[:, :], float32[:, :])],
                     '(m,n),(n,p)->(m,p)',
                     target='cuda')
        def matmulcore(A, B, C):
            m, n = A.shape
            n, p = B.shape
            for i in range(m):
                for j in range(p):
                    C[i, j] = 0
                    for k in range(n):
                        C[i, j] += A[i, k] * B[k, j]

        gufunc = matmulcore
        gufunc.max_blocksize = 512

        matrix_ct = 1001 # an odd number to test thread/block division in CUDA
        A = np.arange(matrix_ct * 2 * 4, dtype=np.float32).reshape(matrix_ct, 2,
                                                                   4)
        B = np.arange(matrix_ct * 4 * 5, dtype=np.float32).reshape(matrix_ct, 4,
                                                                   5)

        gufunc.max_blocksize = 32
        C = gufunc(A, B)
        Gold = ut.matrix_multiply(A, B)
        self.assertTrue(np.allclose(C, Gold))

    def test_gufunc_stream(self):

        @guvectorize([void(float32[:, :], float32[:, :], float32[:, :])],
                     '(m,n),(n,p)->(m,p)',
                     target='cuda')
        def matmulcore(A, B, C):
            m, n = A.shape
            n, p = B.shape
            for i in range(m):
                for j in range(p):
                    C[i, j] = 0
                    for k in range(n):
                        C[i, j] += A[i, k] * B[k, j]

        gufunc = matmulcore
        gufunc.max_blocksize = 512

        #cuda.driver.flush_pending_free()
        matrix_ct = 1001 # an odd number to test thread/block division in CUDA
        A = np.arange(matrix_ct * 2 * 4, dtype=np.float32).reshape(matrix_ct, 2,
                                                                   4)
        B = np.arange(matrix_ct * 4 * 5, dtype=np.float32).reshape(matrix_ct, 4,
                                                                   5)

        ts = time()
        stream = cuda.stream()
        dA = cuda.to_device(A, stream)
        dB = cuda.to_device(B, stream)

        dC = cuda.device_array(shape=(1001, 2, 5), dtype=A.dtype, stream=stream)
        dC = gufunc(dA, dB, out=dC, stream=stream)
        C = dC.copy_to_host(stream=stream)
        stream.synchronize()

        tcuda = time() - ts

        ts = time()
        Gold = ut.matrix_multiply(A, B)
        tcpu = time() - ts

        stream_speedups.append(tcpu / tcuda)

        self.assertTrue(np.allclose(C, Gold))

    def test_copy(self):

        @guvectorize([void(float32[:], float32[:])],
                     '(x)->(x)',
                     target='cuda')
        def copy(A, B):
            for i in range(B.size):
                B[i] = A[i]

        A = np.arange(10, dtype=np.float32) + 1
        B = np.zeros_like(A)
        copy(A, out=B)
        self.assertTrue(np.allclose(A, B))

    def test_copy_odd(self):

        @guvectorize([void(float32[:], float32[:])],
                     '(x)->(x)',
                     target='cuda')
        def copy(A, B):
            for i in range(B.size):
                B[i] = A[i]

        A = np.arange(11, dtype=np.float32) + 1
        B = np.zeros_like(A)
        copy(A, out=B)
        self.assertTrue(np.allclose(A, B))

    def test_copy2d(self):

        @guvectorize([void(float32[:, :], float32[:, :])],
                     '(x, y)->(x, y)',
                     target='cuda')
        def copy2d(A, B):
            for x in range(B.shape[0]):
                for y in range(B.shape[1]):
                    B[x, y] = A[x, y]

        A = np.arange(30, dtype=np.float32).reshape(5, 6) + 1
        B = np.zeros_like(A)
        copy2d(A, out=B)
        self.assertTrue(np.allclose(A, B))


if __name__ == '__main__':
    unittest.main()

