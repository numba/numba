cimport cython

cdef extern from "math.h":
    double sqrt(double x)

@cython.boundscheck(False)
@cython.wraparound(False)
def poly_d(double[::1] A, double[::1] B, double[::1] C, double[::1] D):
    cdef int N = A.shape[0]
    for i in range(N):
        D[i] = sqrt(B[i]**2 + 4 * A[i] * C[i])
        
