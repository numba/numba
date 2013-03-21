# Example from Travis Oliphant

import math
import numpy as np
from numba import jit, autojit
try:
    from numba.tests.support import ctypes_values as rng
except ImportError:
    rng = None

#@jit('double[:,:](int64, int64)')
@autojit
def gibbs(rk_seed, N, thin):
    rk_seed(0, rng.state_p)

    x = 0
    y = 0
    samples = np.empty((N,2))
    for i in range(N):
        for j in range(thin):
            #x = np.random.gamma(3,1.0/(y**2+4))
            x = rng.rk_gamma(rng.state_p, 3.0, 1.0/(y**2+4))
            #y = np.random.normal(1.0/(x+1), 1.0/math.sqrt(2+2*x))
            y = rng.rk_normal(rng.state_p, 1.0/(x+1), 1.0/math.sqrt(2+2*x))

        samples[i, 0] = x
        samples[i, 1] = y

    return samples

def test():
    if rng is not None:
        assert np.allclose(gibbs(rng.rk_seed, 10, 10),
                           gibbs.py_func(rng.rk_seed, 10, 10))

if __name__ == '__main__':
    test()
