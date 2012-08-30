#
# A simple test based on 
#  http://jakevdp.github.com/blog/2012/08/24/numba-vs-cython/
#

from profutils import *
import numpy as np
import numba as nb

import pyximport; pyximport.install()
from cygufunc import *

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


def run_bench():
    signature = [nb.d[:,:], nb.d[:,:]]
    pw_numba = jit(arg_types=signature)(pairwise_python)

    gufunc = GUFuncVectorize(pairwise_python, '(m,n)->(m,m)')
    gufunc.add(arg_types=[nb.d[:,:], nb.d[:,:]])
    pw_numbapro = gufunc.build_ufunc()

    X = np.random.random((1000, 3))
    D = np.zeros((1000, 1000))
    test_args = [ X, D ]

    return profile_functions([('pure_python', pairwise_python, test_args),
                              ('numba', pw_numba, test_args),
                              ('numbapro-gufunc', pw_numbapro, test_args),
                              ('cython', pairwise_cython, test_args)])

if __name__ == '__main__':
    print_profile_results(run_bench())
