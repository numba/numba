'''
Implementation of the Monte Carlo pricer using numbapro.CU 
with heterogeneous targets.
'''

from contextlib import closing, nested
import numpy as np
from math import sqrt, exp
from timeit import default_timer as timer
from numbapro import CU
from numbapro.parallel.kernel import builtins
#from matplotlib import pyplot

def step(tid, paths, dt,  prices, c0, c1, noises):
    paths[tid] = prices[tid] * np.exp(c0 * dt + c1 * noises[tid])

def monte_carlo_pricer(paths, dt, interest, volatility):
    n = paths.shape[0]
    half = n // 2
    cud = CU('gpu')
    cuh = CU('cpu')

    with nested(closing(cud), closing(cuh)):

        cud.configure(builtins.random.seed, 1234)

        c0 = interest - 0.5 * volatility ** 2
        c1 = volatility * np.sqrt(dt)

        d_normdist = cud.scratch(half, dtype=np.double)

        d_last_paths = cud.inout(paths[:half, 0])
        h_last_paths = cuh.input(paths[half:, 0])
        for i in range(1, paths.shape[1]):
            d_paths = cud.inout(paths[:half, i])
            h_paths = cuh.inout(paths[half:, i])

            # generate randomized normal distribution
            cud.enqueue(builtins.random.normal,
                       ntid=half,
                       args=(d_normdist,))
            
            normdist = np.random.normal(size=half)
            h_normdist = cuh.input(normdist)
            # calculate next step

            # GPU
            cud.enqueue(step,
                       ntid=half,
                       args=(d_paths, dt, d_last_paths, c0, c1, d_normdist))

            # CPU
            cuh.enqueue(step,
                        ntid=half,
                        args=(h_paths, dt, h_last_paths, c0, c1, h_normdist))

            d_last_paths = d_paths
            h_last_paths = h_paths


        cud.wait()
        cuh.wait()


if __name__ == '__main__':
    from driver import driver
    driver(monte_carlo_pricer)
