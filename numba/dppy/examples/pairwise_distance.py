from time import time
import numba
from numba import int32, float32
from math import ceil,sqrt
import numpy as np
import argparse
import timeit

from numba import dppy
import dppy.core as ocldrv

parser = argparse.ArgumentParser(description='Program to compute pairwise distance')

parser.add_argument('-n', type=int, required=True, help='Number of points')
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


@dppy.kernel
def pairwise_distance(X, D, xshape0, xshape1):
    idx = dppy.get_global_id(0)

    #for i in range(xshape0):
    for j in range(X.shape[0]):
        d = 0.0
        for k in range(X.shape[1]):
            tmp = X[idx, k] - X[j, k]
            d += tmp * tmp
        D[idx, j] = sqrt(d)


def driver(device_env):
    #measure running time
    times = list()

    # Copy the data to the device
    dX = device_env.copy_array_to_device(X)
    dD = ocldrv.DeviceArray(device_env.get_env_ptr(), D)

    for repeat in range(args.r):
        start = time()
        pairwise_distance[global_size, local_size](dX, dD, X.shape[0], X.shape[1])
        end = time()

        total_time = end - start
        times.append(total_time)

    # Get the data back from device to host
    device_env.copy_array_from_device(dD)

    return times


def main():
    times = None

    if ocldrv.has_gpu_device:
        with ocldrv.igpu_context(0) as device_env:
            times = driver(device_env)
    elif ocldrv.has_cpu_device:
        with ocldrv.cpu_context(0) as device_env:
            times = driver(device_env)
    else:
        print("No device found")
        exit()

    times =  np.asarray(times, dtype=np.float32)
    print("Average time of %d runs is = %fs" % (args.r, times.mean()))


if __name__ == '__main__':
    main()
