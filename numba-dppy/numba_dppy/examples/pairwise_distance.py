from time import time
import numba
from numba import int32, float32
from math import ceil,sqrt
import numpy as np
import argparse
import timeit

from numba import dppl
import dpctl
import dpctl._memory as dpctl_mem

parser = argparse.ArgumentParser(description='Program to compute pairwise distance')

parser.add_argument('-n', type=int, default=10, help='Number of points')
parser.add_argument('-d', type=int, default=3, help='Dimensions')
parser.add_argument('-r', type=int, default=1, help='repeat')
parser.add_argument('-l', type=int, default=1, help='local_work_size')

args = parser.parse_args()

# Global work size is equal to the number of points
global_size = args.n
# Local Work size is optional
local_size = args.l

X = np.random.random((args.n, args.d))
D = np.empty((args.n, args.n))


@dppl.kernel
def pairwise_distance(X, D, xshape0, xshape1):
    idx = dppl.get_global_id(0)

    #for i in range(xshape0):
    for j in range(X.shape[0]):
        d = 0.0
        for k in range(X.shape[1]):
            tmp = X[idx, k] - X[j, k]
            d += tmp * tmp
        D[idx, j] = sqrt(d)


def driver():
    #measure running time
    times = list()

    xbuf = dpctl_mem.MemoryUSMShared(X.size * X.dtype.itemsize)
    x_ndarray = np.ndarray(X.shape, buffer=xbuf, dtype=X.dtype)
    np.copyto(x_ndarray, X)

    dbuf = dpctl_mem.MemoryUSMShared(D.size * D.dtype.itemsize)
    d_ndarray = np.ndarray(D.shape, buffer=dbuf, dtype=D.dtype)
    np.copyto(d_ndarray, D)

    for repeat in range(args.r):
        start = time()
        pairwise_distance[global_size, local_size](x_ndarray, d_ndarray, X.shape[0], X.shape[1])
        end = time()

        total_time = end - start
        times.append(total_time)

    np.copyto(X, x_ndarray)
    np.copyto(D, d_ndarray)

    return times


def main():
    times = None

    if dpctl.has_gpu_queues():
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            times = driver()
    elif dpctl.has_cpu_queues():
        with dpctl.device_context("opencl:cpu") as cpu_queue:
            times = driver()
    else:
        print("No device found")
        exit()

    times =  np.asarray(times, dtype=np.float32)
    print("Average time of %d runs is = %fs" % (args.r, times.mean()))


if __name__ == '__main__':
    main()
