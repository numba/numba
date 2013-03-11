'''
Implementation of the Monte Carlo pricer using numbapro.CU 
with heterogeneous targets.
'''

from contextlib import closing, nested
import numpy as np
from math import sqrt, exp, pi
from timeit import default_timer as timer
from numbapro import CU, uint32, double
from numbapro.parallel.kernel import builtins
#from matplotlib import pyplot

def step(tid, paths, dt,  prices, c0, c1, noises):
    paths[tid] = prices[tid] * np.exp(c0 * dt + c1 * noises[tid])

A = 1664525
C = 1013904223
p0 = 0.322232431088
q0 = 0.099348462606
p1 = 1.0
q1 = 0.588581570495
p2 = 0.342242088547
q2 = 0.531103462366;
p3 = 0.204231210245e-1
q3 = 0.103537752850
p4 = 0.453642210148e-4
q4 = 0.385607006340e-2

def normal(tid, out, seeds):
    seed = seeds[tid]
    randint = (A * seed + C) % 0xfffffff
    seeds[tid] = randint
    u = randint / double(0xfffffff)

    if u < 0.5:
        t = np.sqrt(-2.0 * np.log(u))
    else:
        t = np.sqrt(-2.0 * np.log(1.0 - u))

    p = p0 + t * (p1 + t * (p2 + t * (p3 + t * p4)))
    q = q0 + t * (q1 + t * (q2 + t * (q3 + t * q4)))
    if u < 0.5:
        z = (p / q) - t
    else:
        z = t - (p / q)
    out[tid] = z


def monte_carlo_pricer(paths, dt, interest, volatility):
    n = paths.shape[0]
    half = n // 2
    cud = CU('gpu')
    cuh = CU('cpu')

    seeds = np.random.random_integers(0, 0xffffffff, size=half)

    with nested(closing(cud), closing(cuh)):

        cud.configure(builtins.random.seed, 1234)

        c0 = interest - 0.5 * volatility ** 2
        c1 = volatility * np.sqrt(dt)

        d_normdist = cud.scratch(half, dtype=np.double)
        h_normdist = cuh.scratch(half, dtype=np.double)
        h_seeds = cuh.input(seeds)

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
            cuh.enqueue(normal,
                        ntid=half,
                        args=(h_normdist, h_seeds))
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
