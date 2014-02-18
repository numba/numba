from numba.decorators import jit
from numba import float32, float_
import numpy as np
import numpy.core.umath_tests as ut
from numbapro.vectorizers import GUVectorize
from .support import testcase, main, assertTrue

def matmulcore(A, B, C):
    m, n = A.shape
    n, p = B.shape
    for i in range(m):
        for j in range(p):
            C[i, j] = 0
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]

@testcase
def test_numba():
    jit_matmulcore = jit(argtypes=[float32[:,:], float32[:,:], float32[:,:]])(matmulcore)

    A = np.arange(16, dtype=np.float32).reshape(4, 4)
    B = np.arange(16, dtype=np.float32).reshape(4, 4)
    C = np.zeros(16, dtype=np.float32).reshape(4, 4)
    Gold = np.matrix(A) * np.matrix(B)

    jit_matmulcore(A, B, C)

    if (C != Gold).any():
        raise ValueError

def _test_gufunc(target):
    gufunc = GUVectorize(matmulcore, '(m,n),(n,p)->(m,p)', target=target)
    gufunc.add(argtypes=[float32[:,:], float32[:,:], float32[:,:]])
    gufunc = gufunc.build_ufunc()

    matrix_ct = 1001 # an odd number to test thread/block division in CUDA
    A = np.arange(matrix_ct * 2 * 4, dtype=np.float32).reshape(matrix_ct, 2, 4)
    B = np.arange(matrix_ct * 4 * 5, dtype=np.float32).reshape(matrix_ct, 4, 5)

    C = gufunc(A, B)
    Gold = ut.matrix_multiply(A, B)

    assertTrue(np.allclose(C, Gold))

#
### test gufuncs
#
# def array_expr_gufunc(A, B, C):
#     m, n = A.shape
#     n, p = B.shape
#     for i in range(m):
#         for j in range(p):
#             result = (A[i, :] * B[:, j]).sum()
#             # print result
#             C[i, j] = result
#
# @testcase
# def test_gufunc_array_expressions():
#     gufunc = GUVectorize(array_expr_gufunc, '(m,n),(n,p)->(m,p)')
#     gufunc.add(argtypes=[float_[:,:], float_[:,:], float_[:,:]])
#     gufunc = gufunc.build_ufunc()
#
#     matrix_ct = 10
#     A = np.arange(matrix_ct * 2 * 4, dtype=np.float32).reshape(matrix_ct, 2, 4)
#     B = np.arange(matrix_ct * 4 * 5, dtype=np.float32).reshape(matrix_ct, 4, 5)
#
#     C = gufunc(A, B)
#     Gold = ut.matrix_multiply(A, B)
#
#     if (C != Gold).any():
#         print(C)
#         print(Gold)
#         raise ValueError

@testcase
def test_gufunc():
    _test_gufunc('cpu')



if __name__ == '__main__':
    main()

