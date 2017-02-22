'''
Copyright (c) 2016, Intel Corporation
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
import argparse
import time

parallel = numba.config.NUMBA_NUM_THREADS > 1

@numba.njit(parallel=parallel)
def logistic_regression(labels, points, w, iterations):
    # synthetic but deterministic initial values
    points1 = points.transpose()
    for i in range(iterations):
       y = (1.0+np.exp(-labels*(np.dot(w,points))))
       w -= np.dot(((1.0/y-1.0)*labels), points1)
       #y = (1.0/(1.0+np.exp(-labels*(np.dot(w,points))))-1.0) * labels
       #w -= np.dot(y, points1)

    return w

all_tics = []
def tic():
    global all_tics
    all_tics.append(time.time())

def toq():
    global all_tics
    return (time.time() - all_tics.pop())

def main():
    parser = argparse.ArgumentParser(description='Logistic Regression.')
    parser.add_argument('--dimension', dest='dimension', type=int, default=10)
    parser.add_argument('--points', dest='points', type=int, default=20000000)
    parser.add_argument('--iterations', dest='iterations', type=int, default=30)
    args = parser.parse_args()

    np.random.seed(0)
    D = 3
    N = 4
    iterations = 10
    points = np.ones((D, N))
    labels = np.ones((1, N))
    w = 2.0*np.ones((1,D))-1.3
    tic()
    w = logistic_regression(labels, points, w, iterations)
    compiletime = toq()
    print("SELFPRIMED ", compiletime) 
    print("checksum ", w)
 
    D = args.dimension
    N = args.points
    iterations = args.iterations
    print("D=",D," N=",N," iterations=",iterations)
    points = np.random.random((D, N))
    labels = np.random.random((1, N))
    w = 2.0*np.ones((1,D))-1.3
    tic()
    w = logistic_regression(labels, points, w, iterations)
    selftimed = toq()
    print("SELFTIMED ", selftimed) 
    print("checksum: ", np.sum(w))

main()
 
