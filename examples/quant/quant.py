'''
Copyright (c) 2015, Intel Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are met:
- Redistributions of source code must retain the above copyright notice, 
  this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice, 
  this list of conditions and the following disclaimer in the documentation 
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF 
THE POSSIBILITY OF SUCH DAMAGE.
'''
import numba 
import numpy as np
import numpy.linalg as la
import math
import argparse
import time

#parallel = numba.config.NUMBA_NUM_THREADS > 1
parallel = False

spot = 100.0
strike = 110.0
vol = 0.25
maturity = 2.0

def init(numPaths, numSteps):
    # Pre-compute some values
    dt = maturity / numSteps
    volSqrtdt = vol * math.sqrt(dt)
    fwdFactor = pow(vol, 2) * dt * (-0.5)
    vsqrtdt_log2e = volSqrtdt * math.log(math.e, 2)
    fwdFactor_log2e = fwdFactor * math.log(math.e, 2)
    # Storage
    # Compute all per-path asset values for each time step
    # NOTE: hand-hoisting obscures the basic formula. 
    # Can the compiler do this optimization?
    asset = []
    asset.append(np.full((numPaths,), spot))
    for s in range(numSteps):
        asset.append(asset[s] * pow(2.0, fwdFactor_log2e + vsqrtdt_log2e * np.random.randn(numPaths)))
    return(asset)

#@jit("float64[:](float64, float64[:], float64[:])", nopython=True)
@numba.njit(parallel=parallel)
def model_kernel(strike, curasset, cashFlowPut):
    vcond = curasset <= strike
    valPut0 = curasset * 0.0 # zeros(numPaths)
    valPut0[vcond] = 1.0
    valPut1 = valPut0 * curasset # assetval
    valPut2 = valPut1 * valPut1  # assetval ^ 2
    valPut3 = valPut2 * valPut1  # assetval ^ 3
    valPut4 = valPut2 * valPut2  # assetval ^ 4
    # compute the regression
    ridgecorrect = 0.01
    sum0 = np.sum(valPut0)
    sum1 = np.sum(valPut1)
    sum2 = np.sum(valPut2)
    sum3 = np.sum(valPut3)
    sum4 = np.sum(valPut4)
    sum0 += ridgecorrect
    sum1 += ridgecorrect
    sum2 += ridgecorrect
    sum3 += ridgecorrect
    sum4 += ridgecorrect
    sqrtest = np.empty((3,3))
    sqrtest[0,0] = sum0
    sqrtest[0,1] = sum1
    sqrtest[0,2] = sum2
    sqrtest[1,0] = sum1
    sqrtest[1,1] = sum2
    sqrtest[1,2] = sum3
    sqrtest[2,0] = sum2
    sqrtest[2,1] = sum3
    sqrtest[2,2] = sum4
    invsqr = la.inv(sqrtest)
    vp0 = np.dot(cashFlowPut, valPut0)
    vp1 = np.dot(cashFlowPut, valPut1)
    vp2 = np.dot(cashFlowPut, valPut2)
    vp = np.empty((1,3))
    vp[0,0] = vp0
    vp[0,1] = vp1
    vp[0,2] = vp2
    betaPut = vp @ invsqr
    regImpliedCashFlow = valPut0 * betaPut[0,0] + valPut1 * betaPut[0,1] + valPut2 * betaPut[0,2]
    payoff = valPut0 * (strike - curasset)
    pcond = payoff > regImpliedCashFlow
    cashFlowPut[pcond] = payoff[pcond]
    return cashFlowPut

def model(strike, numPaths, numSteps, asset): 
    avgPathFactor = 1.0/numPaths
    putpayoff = np.maximum(strike - asset[numSteps], 0.0)
    cashFlowPut = putpayoff * avgPathFactor 
    # Now go back in time using regression
    #print(typeof(strike), typeof(asset[0]), typeof(cashFlowPut))
    for s in range(numSteps,1,-1):
        curasset = asset[s]
        cashFloatPut = model_kernel(strike, curasset, cashFlowPut)
    amerpayoffput = sum(cashFlowPut)
    finalputpayoff = sum(putpayoff)
    return amerpayoffput/numPaths, finalputpayoff/numPaths

all_tics = []
def tic():
    global all_tics
    all_tics.append(time.time())

def toq():
    global all_tics
    return (time.time() - all_tics.pop())

def main():
    parser = argparse.ArgumentParser(description='Quantitative option pricing model.')
    parser.add_argument('--assets', dest='assets', type=int, default=524288)
    parser.add_argument('--iterations', dest='iterations', type=int, default=256)
    args = parser.parse_args()
    paths = args.assets 
    steps = args.iterations

    np.random.seed(0)

    print("assets = ", paths)
    print("iterations = ", steps)

    asset = init(paths, steps)
    tic()
    model(strike, 1, 1, asset) 
    compiletime = toq()
    print("SELFPRIMED ", compiletime) 

    tic()
    amerpayoffput, finalputpayoff = model(strike, paths, steps, asset) 
    selftimed = toq()
    print("European Put Option Price: ", finalputpayoff)
    print("American Put Option Price: ", amerpayoffput)
    print("SELFTIMED ", selftimed)

main()
