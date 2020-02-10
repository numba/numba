from time import time
import numba
from numba import int32, float32
from math import ceil,sqrt
import numpy as np
import argparse
import timeit

from numba import dppy
from numba.dppy.dppy_driver import driver as ocldrv

parser = argparse.ArgumentParser(description='Program to compute pairwise distance')

parser.add_argument('-n', type=int, required=True, help='Number of points')
parser.add_argument('-d', type=int, default=3, help='Dimensions')
parser.add_argument('-r', type=int, default=1, help='repeat')
parser.add_argument('-l', type=int, default=1, help='local_work_size')

args = parser.parse_args()

@dppy.kernel
def pairwise_python(X, D, xshape0, xshape1):
    idx = dppy.get_global_id(0)

    #for i in range(xshape0):
    for j in range(X.shape[0]):
        d = 0.0
        for k in range(X.shape[1]):
            tmp = X[idx, k] - X[j, k]
            d += tmp * tmp
        D[idx, j] = sqrt(d)

def call_ocl():
    global_size = args.n
    local_size = args.l

    X = np.random.random((args.n, args.d))
    D = np.empty((args.n, args.n))

    #measure running time
    device_env = None
    device_env = ocldrv.runtime.get_cpu_device()
    start = time()
    times = list()

    dX = device_env.copy_array_to_device(X)
    dD = ocldrv.DeviceArray(device_env.get_env_ptr(), D)

    for repeat in range(args.r):
        start = time()
        pairwise_python[device_env,global_size,local_size](dX, dD, X.shape[0], X.shape[1])
        end = time()

        total_time = end - start
        times.append(total_time)

    device_env.copy_array_from_device(dD)

    times =  np.asarray(times, dtype=np.float32)
    print("Average time of %d runs is = %f" % (args.r, times.mean()))

def main():
    call_ocl()

if __name__ == '__main__':
    main()
