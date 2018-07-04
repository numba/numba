#! /usr/bin/env python
from __future__ import print_function

import math
import time

import numpy as np

from numba import jit

from blackscholes import black_scholes


RISKFREE = 0.02
VOLATILITY = 0.30


@jit
def cnd_numba(d):
    A1 = 0.31938153
    A2 = -0.356563782
    A3 = 1.781477937
    A4 = -1.821255978
    A5 = 1.330274429
    RSQRT2PI = 0.39894228040143267793994605993438
    K = 1.0 / (1.0 + 0.2316419 * math.fabs(d))
    ret_val = (RSQRT2PI * math.exp(-0.5 * d * d) *
               (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))
    if d > 0:
        ret_val = 1.0 - ret_val
    return ret_val


@jit
def black_scholes_numba(callResult, putResult, stockPrice, optionStrike,
                        optionYears, Riskfree, Volatility):
    S = stockPrice
    X = optionStrike
    T = optionYears
    R = Riskfree
    V = Volatility
    for i in range(len(S)):
        sqrtT = math.sqrt(T[i])
        d1 = (math.log(S[i] / X[i]) + (R + 0.5 * V * V) * T[i]) / (V * sqrtT)
        d2 = d1 - V * sqrtT
        cndd1 = cnd_numba(d1)
        cndd2 = cnd_numba(d2)

        expRT = math.exp((-1. * R) * T[i])
        callResult[i] = (S[i] * cndd1 - X[i] * expRT * cndd2)
        putResult[i] = (X[i] * expRT * (1.0 - cndd2) - S[i] * (1.0 - cndd1))


def randfloat(rand_var, low, high):
    return (1.0 - rand_var) * low + rand_var * high


def main (*args):
    OPT_N = 4000000
    iterations = 10
    if len(args) >= 2:
        iterations = int(args[0])

    callResultNumpy = np.zeros(OPT_N)
    putResultNumpy = -np.ones(OPT_N)
    stockPrice = randfloat(np.random.random(OPT_N), 5.0, 30.0)
    optionStrike = randfloat(np.random.random(OPT_N), 1.0, 100.0)
    optionYears = randfloat(np.random.random(OPT_N), 0.25, 10.0)
    callResultNumba = np.zeros(OPT_N)
    putResultNumba = -np.ones(OPT_N)

    time0 = time.time()
    for i in range(iterations):
        black_scholes(callResultNumpy, putResultNumpy, stockPrice,
                      optionStrike, optionYears, RISKFREE, VOLATILITY)
    time1 = time.time()
    print("Numpy Time: %f msec" %
          ((1000 * (time1 - time0)) / iterations))

    time0 = time.time()
    for i in range(iterations):
        black_scholes_numba(callResultNumba, putResultNumba, stockPrice,
                            optionStrike, optionYears, RISKFREE, VOLATILITY)
    time1 = time.time()
    print("Numba Time: %f msec" %
          ((1000 * (time1 - time0)) / iterations))

    delta = np.abs(callResultNumpy - callResultNumba)
    L1norm = delta.sum() / np.abs(callResultNumpy).sum()
    print("L1 norm: %E" % L1norm)
    print("Max absolute error: %E" % delta.max())

if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
