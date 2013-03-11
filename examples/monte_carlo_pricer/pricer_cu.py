'''
Implementation of the Monte Carlo pricer using numbapro.CU with GPU target
'''
from contextlib import closing
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
    cu = CU('gpu')
    
    with closing(cu):
        cu.configure(builtins.random.seed, 1234)

        c0 = interest - 0.5 * volatility ** 2
        c1 = volatility * np.sqrt(dt)

        d_normdist = cu.scratch(n, dtype=np.double)

        d_last_paths = cu.inout(paths[:, 0])
        for i in range(1, paths.shape[1]):
            d_paths = cu.inout(paths[:, i])
            # generate randomized normal distribution
            cu.enqueue(builtins.random.normal,
                       ntid=n,
                       args=(d_normdist,))

            # calculate next step
            cu.enqueue(step,
                       ntid=n,
                       args=(d_paths, dt, d_last_paths, c0, c1, d_normdist))
            d_last_paths = d_paths


        cu.wait()

if __name__ == '__main__':
    from driver import driver
    driver(monte_carlo_pricer)
