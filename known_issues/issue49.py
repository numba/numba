from numba import void, float32
import numpy as np
import numpy.core.umath_tests as ut
from numbapro import guvectorize
from numbapro import cuda
from timeit import default_timer as time

@guvectorize([void(float32[:,:], float32[:,:], float32[:,:])],
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

def test_gufunc_stream():
    matrix_ct = 513  # an odd number to test thread/block division in CUDA
    A = np.arange(matrix_ct * 2 * 4, dtype=np.float32).reshape(matrix_ct, 2, 4)
    B = np.arange(matrix_ct * 4 * 5, dtype=np.float32).reshape(matrix_ct, 4, 5)

    ts = time()
    stream = cuda.stream()

    dC = cuda.device_array(shape=(matrix_ct, 2, 5), dtype=A.dtype, stream=stream)
    dA = cuda.to_device(A, stream)
    dB = cuda.to_device(B, stream)

# NOTE: works if this dC is allocated last
#    dC = cuda.device_array(shape=(matrix_ct, 2, 5), dtype=A.dtype, stream=stream)

    dC = gufunc(dA, dB, out=dC, stream=stream)
    C = dC.copy_to_host(stream=stream)
    stream.synchronize()

    tcuda = time() - ts

    ts = time()
    Gold = ut.matrix_multiply(A, B)
    tcpu = time() - ts

    assert np.allclose(C, Gold)

if __name__ == '__main__':
    test_gufunc_stream()

