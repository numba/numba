#
# Copyright (c) 2017 Intel Corporation
# SPDX-License-Identifier: BSD-2-Clause
#

import numba
import numpy as np
import math
import argparse
import time

run_parallel = numba.config.NUMBA_NUM_THREADS > 1

@numba.vectorize(nopython=True)
def cndf2(inp):
    out = 0.5 + 0.5 * math.erf((math.sqrt(2.0)/2.0) * inp)
    return out

@numba.njit(parallel=run_parallel,fastmath=True)
def blackscholes(sptprice, strike, rate, volatility, timev):
    logterm = np.log(sptprice / strike)
    powterm = 0.5 * volatility * volatility
    den = volatility * np.sqrt(timev)
    d1 = (((rate + powterm) * timev) + logterm) / den
    d2 = d1 - den
    NofXd1 = cndf2(d1)
    NofXd2 = cndf2(d2)
    futureValue = strike * np.exp(- rate * timev)
    c1 = futureValue * NofXd2
    call = sptprice * NofXd1 - c1
    put  = call - futureValue + sptprice
    return put


def run(OPT_N, iterations):
    sptprice   = np.full((OPT_N,), 42.0)
    initStrike = 40 + (np.arange(OPT_N) + 1.0) / OPT_N
    rate       = np.full((OPT_N,), 0.5)
    volatility = np.full((OPT_N,), 0.2)
    timev      = np.full((OPT_N,), 0.5)

    t1 = time.time()
    for i in range(iterations):
        put = blackscholes(sptprice, initStrike, rate, volatility, timev)
    t = time.time()-t1
    return 1000 * (t / iterations)

def main(*args):
    OPT_N = 4000000
    iterations = 10
    if len(args) >= 2:
        iterations = int(args[0])
    
    run(1, 1)
    t = run(OPT_N, iterations)
    print("SELFTIMED ", t)

if __name__ == '__main__':
    main()
