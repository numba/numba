cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def add_d(double[::1] A, double[::1] B, double[::1] C):
    cdef int N = A.shape[0]
    for i in range(N):
        C[i] = A[i] + B[i]
        
