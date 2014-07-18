from __future__ import print_function, absolute_import
import numpy as np
import math
import numba.unittest_support as unittest
from timeit import default_timer as timer
from numba.compiler import compile_isolated, compile_extra, Flags
from numba import types, typing
import numba.array as numbarray


RISKFREE = 0.02
VOLATILITY = 0.30


A1 = 0.31938153
A2 = -0.356563782
A3 = 1.781477937
A4 = -1.821255978
A5 = 1.330274429
RSQRT2PI = 0.39894228040143267793994605993438


def cnd_array(d):
    K = 1.0 / (1.0 + 0.2316419 * np.abs(d))
    ret_val = (RSQRT2PI * np.exp(-0.5 * d * d) *
               (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))
    return np.where(d > 0, 1.0 - ret_val, ret_val)

def cnd_array_deferred(d):
    K = 1.0 / (1.0 + 0.2316419 * numbarray.abs(d))
    ret_val = (RSQRT2PI * numbarray.exp(-0.5 * d * d) *
               (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))
    return numbarray.where(d > 0, 1.0 - ret_val, ret_val)


def cnd(d):
    K = 1.0 / (1.0 + 0.2316419 * math.fabs(d))
    ret_val = (RSQRT2PI * math.exp(-0.5 * d * d) *
               (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))
    if d > 0:
        ret_val = 1.0 - ret_val
    return ret_val


def blackscholes_arrayexpr(stockPrice, optionStrike, optionYears, Riskfree,
                           Volatility):
    S = stockPrice
    X = optionStrike
    T = optionYears
    R = Riskfree
    V = Volatility
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT)
    d2 = d1 - V * sqrtT
    cndd1 = cnd_array(d1)
    cndd2 = cnd_array(d2)

    expRT = np.exp(- R * T)

    callResult = (S * cndd1 - X * expRT * cndd2)
    putResult = (X * expRT * (1.0 - cndd2) - S * (1.0 - cndd1))
    return callResult, putResult


def blackscholes_arrayexpr_jitted(stockPrice, optionStrike, optionYears,
                                  Riskfree, Volatility):
    S = stockPrice
    X = optionStrike
    T = optionYears
    R = Riskfree
    V = Volatility
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT)
    d2 = d1 - V * sqrtT
    cndd1 = cnd_array_jitted(d1)
    cndd2 = cnd_array_jitted(d2)

    expRT = np.exp(- R * T)

    callResult = (S * cndd1 - X * expRT * cndd2)
    putResult = (X * expRT * (1.0 - cndd2) - S * (1.0 - cndd1))
    return callResult, putResult


def blackscholes_deferred(stockPrice, optionStrike, optionYears, Riskfree,
                          Volatility):
    S = stockPrice
    X = optionStrike
    T = optionYears
    R = Riskfree
    V = Volatility
    sqrtT = numbarray.sqrt(T)
    d1 = (numbarray.log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT)
    d2 = d1 - V * sqrtT
    cndd1 = cnd_array_deferred(d1)
    cndd2 = cnd_array_deferred(d2)

    expRT = numbarray.exp(- R * T)

    callResult = (S * cndd1 - X * expRT * cndd2)
    putResult = (X * expRT * (1.0 - cndd2) - S * (1.0 - cndd1))
    return callResult, putResult


def blackscholes_scalar(callResult, putResult, stockPrice, optionStrike,
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
        cndd1 = cnd(d1)
        cndd2 = cnd(d2)

        expRT = math.exp((-1. * R) * T[i])
        callResult[i] = (S[i] * cndd1 - X[i] * expRT * cndd2)
        putResult[i] = (X[i] * expRT * (1.0 - cndd2) - S[i] * (1.0 - cndd1))


def blackscholes_scalar_jitted(callResult, putResult, stockPrice, optionStrike,
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
        cndd1 = cnd_jitted(d1)
        cndd2 = cnd_jitted(d2)

        expRT = math.exp((-1. * R) * T[i])
        callResult[i] = (S[i] * cndd1 - X[i] * expRT * cndd2)
        putResult[i] = (X[i] * expRT * (1.0 - cndd2) - S[i] * (1.0 - cndd1))


def randfloat(rand_var, low, high):
    return (1.0 - rand_var) * low + rand_var * high


class TestBlackScholes(unittest.TestCase):
    def test_array_expr(self):
        flags = Flags()
        flags.set("enable_pyobject")

        global cnd_array_jitted
        cr1 = compile_isolated(cnd_array, args=(), flags=flags)
        cnd_array_jitted = cr1.entry_point
        cr2 = compile_isolated(blackscholes_arrayexpr_jitted, args=(),
                                     flags=flags)
        jitted_bs = cr2.entry_point

        OPT_N = 400
        iterations = 1000


        stockPrice = randfloat(np.random.random(OPT_N), 5.0, 30.0)
        optionStrike = randfloat(np.random.random(OPT_N), 1.0, 100.0)
        optionYears = randfloat(np.random.random(OPT_N), 0.25, 10.0)

        args = stockPrice, optionStrike, optionYears, RISKFREE, VOLATILITY

        ts = timer()
        for i in range(iterations):
            callResultGold, putResultGold = blackscholes_arrayexpr(*args)
        te = timer()
        pytime = te - ts

        ts = timer()
        for i in range(iterations):
            callResultNumba, putResultNumba = jitted_bs(*args)
        te = timer()
        jittime = te - ts


        # Use dummy arrays. We'll inject the actual input arrays later
        x1 = numbarray.Array(name='x1')
        x2 = numbarray.Array(name='x2')
        x3 = numbarray.Array(name='x3')

        args = x1, x2, x3, RISKFREE, VOLATILITY

        # Build the deferred array expressions
        callResultDeferred, putResultDeferred = blackscholes_deferred(*args)

        # Force eval and benchmark vectorize compile time
        ts = timer()
        callResultDeferredX = callResultDeferred.eval(x1=stockPrice, x2=optionStrike, x3=optionYears)
        putResultDeferredX = putResultDeferred.eval(x1=stockPrice, x2=optionStrike, x3=optionYears)
        te = timer()
        deferredbuildtime = te - ts

        # Now that vectorize function is already compiled, run calculations
        # for benchmarking
        ts = timer()
        for i in range(iterations):
            callResultDeferredX = callResultDeferred.eval(x1=stockPrice, x2=optionStrike, x3=optionYears)
            putResultDeferredX = putResultDeferred.eval(x1=stockPrice, x2=optionStrike, x3=optionYears)
        te = timer()
        deferredtime = te - ts


        print("Python", pytime)
        print("Numba", jittime)
        print("Speedup: %s" % (pytime / jittime))
        print("Deferred", deferredtime)
        print("Deferred Build", deferredbuildtime)
        print("Speedup: %s" % (pytime / deferredtime))

        delta = np.abs(callResultGold - callResultNumba)
        L1norm = delta.sum() / np.abs(callResultGold).sum()
        print("Array Expr L1 norm: %E" % L1norm)
        print("Array Expr Max absolute error: %E" % delta.max())
        self.assertEqual(delta.max(), 0)

        delta = np.abs(callResultGold - callResultDeferredX)
        L1norm = delta.sum() / np.abs(callResultGold).sum()
        print("Deferred L1 norm: %E" % L1norm)
        print("Deferred Max absolute error: %E" % delta.max())
        self.assertEqual(delta.max(), 0)

if __name__ == "__main__":
    unittest.main()
