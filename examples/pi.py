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
def calc_pi(n):
    x = 2*np.random.ranf(n)-1
    y = 2*np.random.ranf(n)-1
    return 4*np.sum(x**2+y**2<1)/n

def main():
    parser = argparse.ArgumentParser(description='Calculate Pi.')
    parser.add_argument('--points', dest='points', type=int, default=20000000)
    args = parser.parse_args()
    points = args.points
    np.random.seed(0)

    t1 = time.time()
    pi = calc_pi(points)
    selftimed = time.time()-t1
    print("SELFTIMED ", selftimed)
    print("result: ", pi)

if __name__ == '__main__':
    main()
