import numba 
import numpy as np
# from scipy.special import erf
import math
import argparse
import time

parallel = numba.config.NUMBA_NUM_THREADS > 1

@numba.vectorize(nopython=True)
def cndf2(inp):
    out = 0.5 + 0.5 * math.erf(0.707106781 * inp)
    return out

#@numba.njit(parallel=parallel)
@numba.vectorize('f8(f8,f8,f8,f8,f8)', target="parallel")
def blackscholes(sptprice, strike, rate, volatility, time):
    logterm = np.log10(sptprice / strike)
    powterm = 0.5 * volatility * volatility
    den = volatility * np.sqrt(time)
    d1 = (((rate + powterm) * time) + logterm) / den
    d2 = d1 - den
    NofXd1 = cndf2(d1)
    NofXd2 = cndf2(d2)
    futureValue = strike * np.exp(- rate * time)
    c1 = futureValue * NofXd2
    call = sptprice * NofXd1 - c1
    put  = call - futureValue + sptprice
    return put

all_tics = []
def tic():
    global all_tics
    all_tics.append(time.time())

def toq():
    global all_tics
    return (time.time() - all_tics.pop())


def run(iterations):
    sptprice   = np.full((iterations,), 42.0)
    initStrike = 40 + (np.arange(iterations) + 1.0) / iterations
    rate       = np.full((iterations,), 0.5)
    volatility = np.full((iterations,), 0.2)
    time       = np.full((iterations,), 0.5)

    tic()
    put = blackscholes(sptprice, initStrike, rate, volatility, time)
    t = toq()
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

main()
