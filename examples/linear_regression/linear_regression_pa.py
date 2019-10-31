#
# Copyright (c) 2017 Intel Corporation
# SPDX-License-Identifier: BSD-2-Clause
#

import numba
import numpy as np
import argparse
import time

run_parallel = numba.config.NUMBA_NUM_THREADS > 1

@numba.njit(parallel=run_parallel)
def linear_regression(Y, X, w, iterations, alphaN):
    for i in range(iterations):
        w -= alphaN * np.dot(X.T, np.dot(X,w)-Y)
    return w

def main():
    parser = argparse.ArgumentParser(description='Linear Regression.')
    parser.add_argument('--samples', dest='samples', type=int, default=200000)
    parser.add_argument('--features', dest='features', type=int, default=10)
    parser.add_argument('--functions', dest='functions', type=int, default=4)
    parser.add_argument('--iterations', dest='iterations', type=int, default=20)
    args = parser.parse_args()
    N = args.samples
    D = args.features
    p = args.functions
    iterations = args.iterations
    alphaN = 0.01/N
    w = np.zeros((D,p))
    np.random.seed(0)
    points = np.random.random((N,D))
    labels = np.random.random((N,p))
    t1 = time.time()
    w = linear_regression(labels, points, w, iterations, alphaN)
    selftimed = time.time()-t1
    print("SELFTIMED ", selftimed)
    print("checksum: ", np.sum(w))

if __name__ == '__main__':
    main()
