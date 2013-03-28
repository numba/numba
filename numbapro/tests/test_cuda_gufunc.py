from numba.decorators import jit
from numba import *
import numpy as np
import numpy.core.umath_tests as ut
from numbapro.vectorize import GUVectorize
from numbapro import cuda
from timeit import default_timer as time

def matmulcore(A, B, C):
    m, n = A.shape
    n, p = B.shape
    for i in range(m):
        for j in range(p):
            C[i, j] = 0
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]

gufunc = GUVectorize(matmulcore, '(m,n),(n,p)->(m,p)', target='gpu')
gufunc.add(argtypes=[f4[:,:], f4[:,:], f4[:,:]])
gufunc = gufunc.build_ufunc()
gufunc.max_blocksize = 512

non_stream_speedups = []
stream_speedups = []

def _test_gufunc():
    matrix_ct = 1001 # an odd number to test thread/block division in CUDA
    A = np.arange(matrix_ct * 2 * 4, dtype=np.float32).reshape(matrix_ct, 2, 4)
    B = np.arange(matrix_ct * 4 * 5, dtype=np.float32).reshape(matrix_ct, 4, 5)

    ts = time()
    C = gufunc(A, B)
    tcuda = time() - ts

    ts = time()
    Gold = ut.matrix_multiply(A, B)
    tcpu = time() - ts

    non_stream_speedups.append(tcpu / tcuda)
    assert np.allclose(C, Gold)


def _test_gufunc_adjust_blocksize():
    matrix_ct = 1001 # an odd number to test thread/block division in CUDA
    A = np.arange(matrix_ct * 2 * 4, dtype=np.float32).reshape(matrix_ct, 2, 4)
    B = np.arange(matrix_ct * 4 * 5, dtype=np.float32).reshape(matrix_ct, 4, 5)

    ts = time()
    gufunc.max_blocksize = 32
    C = gufunc(A, B)
    tcuda = time() - ts

    ts = time()
    Gold = ut.matrix_multiply(A, B)
    tcpu = time() - ts
    
    assert np.allclose(C, Gold)


def _test_gufunc_stream():
    matrix_ct = 1001 # an odd number to test thread/block division in CUDA
    A = np.arange(matrix_ct * 2 * 4, dtype=np.float32).reshape(matrix_ct, 2, 4)
    B = np.arange(matrix_ct * 4 * 5, dtype=np.float32).reshape(matrix_ct, 4, 5)

    ts = time()
    stream = cuda.stream()
    dA = cuda.to_device(A, stream)
    dB = cuda.to_device(B, stream)
    C = gufunc(dA, dB, stream=stream)
    C.to_host(stream=stream)
    stream.synchronize()
    tcuda = time() - ts

    ts = time()
    Gold = ut.matrix_multiply(A, B)
    tcpu = time() - ts

    stream_speedups.append(tcpu / tcuda)
    assert np.allclose(C, Gold)

def test_cuda_gufunc():
    for _ in range(50):
        _test_gufunc()
        _test_gufunc_adjust_blocksize()
        _test_gufunc_stream()
    print 'CUDA speedup: %f' % max(non_stream_speedups)
    print 'CUDA streamed speedup: %f' % max(stream_speedups)

def main():
    test_cuda_gufunc()
    print 'ok'

if __name__ == '__main__':
    main()

