import numpy as np
import math
import time
from numbapro import cuda, double
from .support import testcase, main

RISKFREE = 0.02
VOLATILITY = 0.30


A1 = 0.31938153
A2 = -0.356563782
A3 = 1.781477937
A4 = -1.821255978
A5 = 1.330274429
RSQRT2PI = 0.39894228040143267793994605993438

@cuda.jit(argtypes=(double,), restype=double, device=True, inline=True)
def cnd_cuda(d):
    K = 1.0 / (1.0 + 0.2316419 * math.fabs(d))
    ret_val = (RSQRT2PI * math.exp(-0.5 * d * d) *
               (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))
    if d > 0:
        ret_val = 1.0 - ret_val
    return ret_val


@cuda.jit(argtypes=(double[:], double[:], double[:], double[:], double[:],
                    double, double))
def black_scholes_cuda(callResult, putResult, S, X, T, R, V):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if i >= S.shape[0]:
        return
    sqrtT = math.sqrt(T[i])
    d1 = (math.log(S[i] / X[i]) + (R + 0.5 * V * V) * T[i]) / (V * sqrtT)
    d2 = d1 - V * sqrtT
    cndd1 = cnd_cuda(d1)
    cndd2 = cnd_cuda(d2)

    expRT = math.exp((-1. * R) * T[i])
    callResult[i] = (S[i] * cndd1 - X[i] * expRT * cndd2)
    putResult[i] = (X[i] * expRT * (1.0 - cndd2) - S[i] * (1.0 - cndd1))


def randfloat(rand_var, low, high):
    return (1.0 - rand_var) * low + rand_var * high

@testcase
def test_blackscholes():
    OPT_N = 400
    iterations = 2

    callResultNumpy = np.zeros(OPT_N)
    putResultNumpy = -np.ones(OPT_N)
    stockPrice = randfloat(np.random.random(OPT_N), 5.0, 30.0)
    optionStrike = randfloat(np.random.random(OPT_N), 1.0, 100.0)
    optionYears = randfloat(np.random.random(OPT_N), 0.25, 10.0)
    callResultNumba = np.zeros(OPT_N)
    putResultNumba = -np.ones(OPT_N)
    callResultNumbapro = np.zeros(OPT_N)
    putResultNumbapro = -np.ones(OPT_N)


    time0 = time.time()
    blockdim = 1024, 1
    griddim = int(math.ceil(float(OPT_N)/blockdim[0])), 1
    stream = cuda.stream()
    d_callResult = cuda.to_device(callResultNumbapro, stream)
    d_putResult = cuda.to_device(putResultNumbapro, stream)
    d_stockPrice = cuda.to_device(stockPrice, stream)
    d_optionStrike = cuda.to_device(optionStrike, stream)
    d_optionYears = cuda.to_device(optionYears, stream)
    time1 = time.time()
    for i in range(iterations):
        black_scholes_cuda[griddim, blockdim, stream](
            d_callResult, d_putResult, d_stockPrice, d_optionStrike,
            d_optionYears, RISKFREE, VOLATILITY)
        d_callResult.to_host(stream)
        d_putResult.to_host(stream)
        stream.synchronize()
    time2 = time.time()
    dt = (time1 - time0) * 10 + (time2 - time1)
    print("numbapro.cuda time: %f msec" % ((1000 * dt) / iterations))

if __name__ == '__main__':
    main()
