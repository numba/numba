cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def copy_d(double[::1] src, double[::1] dst):
    cdef int N = src.shape[0]
    for i in range(N):
        dst[i] = src[i]
        
