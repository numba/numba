#
# A simple test based on 
#  http://jakevdp.github.com/blog/2012/08/24/numba-vs-cython/
#

import time
import numpy as np
import numba as nb

from numba.decorators import jit
from numbapro.vectorize.gufunc import GUFuncVectorize

def pairwise_python(X, D):
    M = X.shape[0]
    N = X.shape[1]
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = np.sqrt(d)

def test_suite(Tests, times = 10):
    X = np.random.random((1000, 3))
    D = np.zeros((1000, 1000))
    def timeit_func(func):
        t0 = time.time()
        func(X,D)
        t1 = time.time()
        return t1-t0
    
    for test in Tests:
        timings = [timeit_func(test[1]) for i in range(0, times)];
        print '%s took avg: %f ms max: %f ms min: %f ms.' %(test[0], 1e+3*sum(timings)/len(timings), 1e+3 * max(timings), 1e+3*min(timings))


def run_bench():
    signature = [nb.double[:,:], nb.double[:,:]]
    pairwise_numba = jit(arg_types=signature)(pairwise_python)

    gufunc = GUFuncVectorize(pairwise_python, '(m,n)->(m,m)')
    gufunc.add(arg_types=[nb.double[:,:], nb.double[:,:]])
    pairwise_numbapro = gufunc.build_ufunc()

    test_suite([('pure python', pairwise_python),
                ('numba', pairwise_numba),
                ('numbapro_gufunc', pairwise_numbapro)]);

if __name__ == '__main__':
    run_bench()
