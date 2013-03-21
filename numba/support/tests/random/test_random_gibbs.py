# Example by Travis Oliphant

try:
    from numba import jit, random
except ImportError:
    pass
else:
    import numpy as np
    import math

    state = random.state_p

    @jit('f8[:,:](int64, int32)')
    def gibbs(N, thin):
        x = 0
        y = 0
        samples = np.empty((N,2))
        for i in range(N):
            for j in range(thin):
                x = random.rk_gamma(state, 3, 1.0/(y**2+4))
                y = random.rk_normal(state, 1.0/(x+1), 1.0/math.sqrt(2+2*x))

            samples[i, 0] = x
            samples[i, 1] = y

        return samples

