import numpy as np
import math
from timeit import default_timer as timer

import numbapro 
from numba import autojit, jit, double, void, uint32

# PRNG constants
A = 1664525
C = 1013904223

@jit(void(double[:,:], double, double, double, double[:], uint32[:]))
def monte_carlo_pricer(paths, dt, interest, volatility, normdist, seed):
    # XXX: prange does not work for some reason with the anaconda release
    #      because py_modulo cannot be linked
    for j in xrange(1, paths.shape[1]):         # foreach time step
        for i in xrange(paths.shape[0]):        # foreach path
            c0 = interest - 0.5 * volatility ** 2
            c1 = volatility * math.sqrt(dt)
            noise = normdist[seed[i] % normdist.shape[0]]

            # XXX: the follow code generates the wrong result
            #    elt = seed[i] - (seed[i] // normdist.shape[0]) * normdist.shape[0]
            #    print elt, seed[i] % normdist.shape[0]
            # XXX: bitwise workaround is not supported
            #    noise = normdist[seed[i] & 0xfff]

            paths[i, j] = paths[i, j - 1] * math.exp(c0 * dt + c1 * noise)
            # generate next random number
            seed[i] = seed[i] * A + C

if __name__ == '__main__':
    from driver2 import driver
    driver(monte_carlo_pricer)
