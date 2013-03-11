'''
Implementation of the Monte Carlo pricer using numbapro.CU 
with heterogeneous targets.
'''

from contextlib import closing, nested
import numpy as np
from math import sqrt, exp, pi
from timeit import default_timer as timer
from numbapro import CU, uint32
from numbapro.parallel.kernel import builtins
#from matplotlib import pyplot

def step(tid, paths, dt,  prices, c0, c1, noises):
    paths[tid] = prices[tid] * np.exp(c0 * dt + c1 * noises[tid])

A = 1664525
C = 1013904223
def step_cpu(tid, paths, dt,  prices, c0, c1, seeds):
    seed = uint32(seeds[tid])
    rand = uint32(A) * seed + uint32(C)
    seeds[tid] = rand
    x = 4. / rand
    noise = 1. / np.sqrt(2 * pi) * np.exp(-(x * x / 2.))
    paths[tid] = prices[tid] * np.exp(c0 * dt + c1 * noise)

def monte_carlo_pricer(paths, dt, interest, volatility):
    n = paths.shape[0]
    half = n // 2
    cud = CU('gpu')
    cuh = CU('cpu')

    seeds = np.random.random(n)


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
            h_seeds = cuh.input(seeds)
            # calculate next step
            # GPU
            cud.enqueue(step,
                       ntid=half,
                       args=(d_paths, dt, d_last_paths, c0, c1, d_normdist))

            # CPU
            cuh.enqueue(step_cpu,
                        ntid=half,
                        args=(h_paths, dt, h_last_paths, c0, c1, h_seeds))

            d_last_paths = d_paths
            h_last_paths = h_paths


        cud.wait()
        cuh.wait()


if __name__ == '__main__':
    from driver import driver
    driver(monte_carlo_pricer)
