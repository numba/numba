from numba.decorators import jit
from numba import *
import numpy as np
import numpy.core.umath_tests as ut
from numbapro.vectorize.gufunc import GUFuncVectorize, CUDAGUFuncVectorize


def matmulcore(A, B, C):
    m, n = A.shape
    n, p = B.shape
    for i in range(m):
        for j in range(p):
            C[i, j] = 0
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]



def test_numba():
    from itertools import product
    jit_matmulcore = jit(argtypes=[f[:,:], f[:,:], f[:,:]])(matmulcore)

    A = np.arange(16, dtype=np.float32).reshape(4, 4)
    B = np.arange(16, dtype=np.float32).reshape(4, 4)
    C = np.zeros(16, dtype=np.float32).reshape(4, 4)
    Gold = np.matrix(A) * np.matrix(B)

    jit_matmulcore(A, B, C)

    if (C != Gold).any():
        raise ValueError

def _test_gufunc(vectorizer):
    gufunc = vectorizer(matmulcore, '(m,n),(n,p)->(m,p)')
    gufunc.add(argtypes=[f[:,:], f[:,:], f[:,:]])
    gufunc = gufunc.build_ufunc()

    matrix_ct = 1001 # an odd number to test thread/block division in CUDA
    A = np.arange(matrix_ct * 2 * 4, dtype=np.float32).reshape(matrix_ct, 2, 4)
    B = np.arange(matrix_ct * 4 * 5, dtype=np.float32).reshape(matrix_ct, 4, 5)

    C = gufunc(A, B)
    Gold = ut.matrix_multiply(A, B)

    # print(C)
    # print(Gold)

    for i, (got, expect) in enumerate(zip(C.flatten(), Gold.flatten())):    
        '''
        CUDA floating point arithmetic has different rounding errors than
        intel/amd cpu. We cannot use `==` to compare.  Instead,
        compute the relative error and expect that to be no greater than 
        a certain threshold (1e-6 in this test).
        '''
        error = (got - expect) / expect
        if error > 1e-6:
            raise ValueError(i, got, expect)


def test_gufunc():
    _test_gufunc(GUFuncVectorize)

def test_cuda_gufunc():
    _test_gufunc(CUDAGUFuncVectorize)

def main():
#    test_numba()
#    test_gufunc()
    try:
        from numbapro import _cudadispatch
    except ImportError:
        print 'skipped CUDA gufunc test'
    else:
        test_cuda_gufunc()

        # stress test
#        for i in range(100):
#            test_gufunc()
#            test_cuda_gufunc()
    print 'All good!'

if __name__ == '__main__':
    main()

