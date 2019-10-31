from __future__ import print_function, absolute_import, division

import numpy as np
import math
import time
from numba import ocl, double
from numba.ocl.testing import unittest


RISKFREE = 0.02
VOLATILITY = 0.30

A1 = 0.31938153
A2 = -0.356563782
A3 = 1.781477937
A4 = -1.821255978
A5 = 1.330274429
RSQRT2PI = 0.39894228040143267793994605993438


def cnd(d):
    K = 1.0 / (1.0 + 0.2316419 * np.abs(d))
    ret_val = (RSQRT2PI * np.exp(-0.5 * d * d) *
               (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))
    return np.where(d > 0, 1.0 - ret_val, ret_val)


def black_scholes(callResult, putResult, stockPrice, optionStrike, optionYears,
                  Riskfree, Volatility):
    S = stockPrice
    X = optionStrike
    T = optionYears
    R = Riskfree
    V = Volatility
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT)
    d2 = d1 - V * sqrtT
    cndd1 = cnd(d1)
    cndd2 = cnd(d2)

    expRT = np.exp(- R * T)
    callResult[:] = (S * cndd1 - X * expRT * cndd2)
    putResult[:] = (X * expRT * (1.0 - cndd2) - S * (1.0 - cndd1))


def randfloat(rand_var, low, high):
    return (1.0 - rand_var) * low + rand_var * high


class TestBlackScholes(unittest.TestCase):
    def test_blackscholes(self):
        OPT_N = 400
        iterations = 2

        stockPrice = randfloat(np.random.random(OPT_N), 5.0, 30.0)
        optionStrike = randfloat(np.random.random(OPT_N), 1.0, 100.0)
        optionYears = randfloat(np.random.random(OPT_N), 0.25, 10.0)

        callResultNumpy = np.zeros(OPT_N)
        putResultNumpy = -np.ones(OPT_N)

        callResultNumbapro = np.zeros(OPT_N)
        putResultNumbapro = -np.ones(OPT_N)

        # numpy
        for i in range(iterations):
            black_scholes(callResultNumpy, putResultNumpy, stockPrice,
                          optionStrike, optionYears, RISKFREE, VOLATILITY)



        @ocl.jit
        def cnd_ocl(d):
            K = 1.0 / (1.0 + 0.2316419 * math.fabs(d))
            ret_val = (RSQRT2PI * math.exp(-0.5 * d * d) *
                       (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))
            if d > 0:
                ret_val = 1.0 - ret_val
            return ret_val


        @ocl.jit
        def black_scholes_ocl(callResult, putResult, S, X, T, R, V):
            i = ocl.get_global_id(0)
            if i >= S.shape[0]:
                return
            sqrtT = math.sqrt(T[i])
            d1 = (math.log(S[i] / X[i]) + (R + 0.5 * V * V) * T[i]) / (V * sqrtT)
            d2 = d1 - V * sqrtT

            K = 1.0 / (1.0 + 0.2316419 * math.fabs(d1))
            cndd1 = (RSQRT2PI * math.exp(-0.5 * d1 * d1) *
                    (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))
            if d1 > 0:
                cndd1 = 1.0 - cndd1

            K = 1.0 / (1.0 + 0.2316419 * math.fabs(d2))
            cndd2 = (RSQRT2PI * math.exp(-0.5 * d2 * d2) *
                    (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))
            if d2 > 0:
                cndd2 = 1.0 - cndd2

            expRT = math.exp((-1. * R) * T[i])
            callResult[i] = (S[i] * cndd1 - X[i] * expRT * cndd2)
            putResult[i] = (X[i] * expRT * (1.0 - cndd2) - S[i] * (1.0 - cndd1))

        # numbapro
        time0 = time.time()
        blockdim = 512, 1
        griddim = int(math.ceil(float(OPT_N) / blockdim[0])), 1
        stream = ocl.stream()
        d_callResult = ocl.to_device(callResultNumbapro, stream)
        d_putResult = ocl.to_device(putResultNumbapro, stream)
        d_stockPrice = ocl.to_device(stockPrice, stream)
        d_optionStrike = ocl.to_device(optionStrike, stream)
        d_optionYears = ocl.to_device(optionYears, stream)
        time1 = time.time()
        for i in range(iterations):
            black_scholes_ocl[griddim, blockdim, stream](
                d_callResult, d_putResult, d_stockPrice, d_optionStrike,
                d_optionYears, RISKFREE, VOLATILITY)
        d_callResult.copy_to_host(callResultNumbapro, stream)
        d_putResult.copy_to_host(putResultNumbapro, stream)
        stream.finish()

        dt = (time1 - time0)

        delta = np.abs(callResultNumpy - callResultNumbapro)
        L1norm = delta.sum() / np.abs(callResultNumpy).sum()

        max_abs_err = delta.max()
        self.assertTrue(L1norm < 1e-13)
        self.assertTrue(max_abs_err < 1e-13)


if __name__ == '__main__':
    unittest.main()
