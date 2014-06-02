from __future__ import print_function, absolute_import
from numba import void, float32
import numpy as np
import numpy.core.umath_tests as ut
from numbapro import guvectorize
from numbapro import cuda
from timeit import default_timer as time
from numbapro.testsupport import unittest


@guvectorize([void(float32[:, :], float32[:, :], float32[:, :])],
             '(m,n),(n,p)->(m,p)',
             target='gpu')
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

non_stream_speedups = []
stream_speedups = []


@guvectorize([void(float32[:], float32[:])],
             '(x)->(x)',
             target='gpu')
def copy(A, B):
    for i in range(B.size):
        B[i] = A[i]


@guvectorize([void(float32[:, :], float32[:, :])],
             '(x, y)->(x, y)',
             target='gpu')
def copy2d(A, B):
    for x in range(B.shape[0]):
        for y in range(B.shape[1]):
            B[x, y] = A[x, y]


class TestCUDAGufunc(unittest.TestCase):
    def test_gufunc_small(self):
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

    def test_gufunc(self):
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
        A = np.arange(10, dtype=np.float32) + 1
        B = np.zeros_like(A)
        copy(A, out=B)
        self.assertTrue(np.allclose(A, B))

    def test_copy_odd(self):
        A = np.arange(11, dtype=np.float32) + 1
        B = np.zeros_like(A)
        copy(A, out=B)
        self.assertTrue(np.allclose(A, B))

    def test_copy2d(self):
        A = np.arange(30, dtype=np.float32).reshape(5, 6) + 1
        B = np.zeros_like(A)
        copy2d(A, out=B)
        self.assertTrue(np.allclose(A, B))


if __name__ == '__main__':
    unittest.main()

