#
# Copyright (c) 2017 Intel Corporation
# SPDX-License-Identifier: BSD-2-Clause
#

import numba
import numpy as np
import math
import argparse
import time

@numba.vectorize(nopython=True)
def cndf2(inp):
    out = 0.5 + 0.5 * math.erf((math.sqrt(2.0)/2.0) * inp)
    return out

@numba.vectorize('f8(f8,f8,f8,f8,f8)', target="parallel")
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

def run(iterations):
    sptprice   = np.full((iterations,), 42.0)
    initStrike = 40 + (np.arange(iterations) + 1.0) / iterations
    rate       = np.full((iterations,), 0.5)
    volatility = np.full((iterations,), 0.2)
    timev       = np.full((iterations,), 0.5)

    t1 = time.time()
    put = blackscholes(sptprice, initStrike, rate, volatility, timev)
    t = time.time()-t1
    print("checksum: ", sum(put))
    print("SELFTIMED ", t)

def main():
    parser = argparse.ArgumentParser(description='Black-Scholes')
    parser.add_argument('--options', dest='options', type=int, default=10000000)
    args = parser.parse_args()
    options = args.options

    run(10)
    print("options = ", options)
    run(options)

if __name__ == '__main__':
    main()
