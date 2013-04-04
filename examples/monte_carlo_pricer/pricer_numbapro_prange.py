import numpy as np
import math
from timeit import default_timer as timer
from numbapro import jit

# PRNG constants
A = 1664525
C = 1013904223

@jit('void(double[:,:], double, double, double, double[:], uint32[:])')
def monte_carlo_pricer(paths, dt, interest, volatility, normdist, seed):
    for j in prange(1, paths.shape[1]):         # foreach time step
        for i in xrange(paths.shape[0]):        # foreach path
            c0 = interest - 0.5 * volatility ** 2
            c1 = volatility * math.sqrt(dt)
            noise = normdist[seed[i] % normdist.shape[0]]
            paths[i, j] = paths[i, j - 1] * math.exp(c0 * dt + c1 * noise)
            # generate next random number
            seed[i] = seed[i] * A + C

if __name__ == '__main__':
    from driver2 import driver
    driver(monte_carlo_pricer)
