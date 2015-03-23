from __future__ import print_function

import math
from timeit import default_timer as timer

import numpy as np

import numba.unittest_support as unittest
from numba.compiler import compile_isolated, compile_extra, Flags
from numba import types, typing
from .support import TestCase


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


class TestBlackScholes(TestCase):
    def test_array_expr(self):
        flags = Flags()
        flags.set("enable_pyobject")

        global cnd_array_jitted
        scalty = types.float64
        arrty = types.Array(scalty, 1, 'C')
        cr1 = compile_isolated(cnd_array, args=(arrty,), flags=flags)
        cnd_array_jitted = cr1.entry_point
        cr2 = compile_isolated(blackscholes_arrayexpr_jitted,
                               args=(arrty, arrty, arrty, scalty, scalty),
                               flags=flags)
        jitted_bs = cr2.entry_point

        OPT_N = 400
        iterations = 10


        stockPrice = randfloat(self.random.random_sample(OPT_N), 5.0, 30.0)
        optionStrike = randfloat(self.random.random_sample(OPT_N), 1.0, 100.0)
        optionYears = randfloat(self.random.random_sample(OPT_N), 0.25, 10.0)

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

        print("Python", pytime)
        print("Numba", jittime)
        print("Speedup: %s" % (pytime / jittime))

        delta = np.abs(callResultGold - callResultNumba)
        L1norm = delta.sum() / np.abs(callResultGold).sum()
        print("L1 norm: %E" % L1norm)
        print("Max absolute error: %E" % delta.max())
        self.assertEqual(delta.max(), 0)

    def test_scalar(self):
        flags = Flags()

        # Compile the inner function
        global cnd_jitted
        cr1 = compile_isolated(cnd, (types.float64,))
        cnd_jitted = cr1.entry_point
        # Manually type the compiled function for calling into
        tyctx = cr1.typing_context
        ctx = cr1.target_context
        signature = typing.make_concrete_template("cnd_jitted", cnd_jitted,
                                                  [cr1.signature])
        tyctx.insert_user_function(cnd_jitted, signature)

        # Compile the outer function
        array = types.Array(types.float64, 1, 'C')
        argtys = (array,) * 5 + (types.float64, types.float64)
        cr2 = compile_extra(tyctx, ctx, blackscholes_scalar_jitted,
                            args=argtys, return_type=None, flags=flags,
                            locals={})
        jitted_bs = cr2.entry_point

        OPT_N = 400
        iterations = 10

        callResultGold = np.zeros(OPT_N)
        putResultGold = np.zeros(OPT_N)

        callResultNumba = np.zeros(OPT_N)
        putResultNumba = np.zeros(OPT_N)

        stockPrice = randfloat(self.random.random_sample(OPT_N), 5.0, 30.0)
        optionStrike = randfloat(self.random.random_sample(OPT_N), 1.0, 100.0)
        optionYears = randfloat(self.random.random_sample(OPT_N), 0.25, 10.0)

        args = stockPrice, optionStrike, optionYears, RISKFREE, VOLATILITY

        ts = timer()
        for i in range(iterations):
             blackscholes_scalar(callResultGold, putResultGold, *args)
        te = timer()
        pytime = te - ts

        ts = timer()
        for i in range(iterations):
            jitted_bs(callResultNumba, putResultNumba, *args)
        te = timer()
        jittime = te - ts

        print("Python", pytime)
        print("Numba", jittime)
        print("Speedup: %s" % (pytime / jittime))

        delta = np.abs(callResultGold - callResultNumba)
        L1norm = delta.sum() / np.abs(callResultGold).sum()
        print("L1 norm: %E" % L1norm)
        print("Max absolute error: %E" % delta.max())
        self.assertAlmostEqual(delta.max(), 0)


if __name__ == "__main__":
    unittest.main()
