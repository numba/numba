'''
Implementation of the Monte Carlo pricer using numbapro.CU with GPU target.
Enhance with two streams.
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

    cu1 = CU('gpu')
    cu2 = CU('gpu')

    with nested(closing(cu1), closing(cu2)):
        cu1.configure(builtins.random.seed, 1234)
        cu2.configure(builtins.random.seed, 4321)

        c0 = interest - 0.5 * volatility ** 2
        c1 = volatility * np.sqrt(dt)

        d_normdist1 = cu1.scratch(half, dtype=np.double)
        d_normdist2 = cu2.scratch(half, dtype=np.double)

        d_last_paths1 = cu1.inout(paths[:half, 0])
        d_last_paths2 = cu2.inout(paths[half:, 0])
        for i in range(1, paths.shape[1]):
            d_paths1 = cu1.inout(paths[:half, i])
            d_paths2 = cu2.inout(paths[half:, i])

            # generate randomized normal distribution
            cu1.enqueue(builtins.random.normal,
                        ntid=half,
                        args=(d_normdist1,))
            cu2.enqueue(builtins.random.normal,
                        ntid=half,
                        args=(d_normdist2,))

            # calculate next step
            cu1.enqueue(step,
                        ntid=half,
                        args=(d_paths1, dt, d_last_paths1, c0, c1, d_normdist1))
            cu2.enqueue(step,
                        ntid=half,
                        args=(d_paths2, dt, d_last_paths2, c0, c1, d_normdist2))
            d_last_paths1 = d_paths1
            d_last_paths2 = d_paths2

        cu1.wait()
        cu2.wait()

if __name__ == '__main__':
    from driver import driver
    driver(monte_carlo_pricer)
