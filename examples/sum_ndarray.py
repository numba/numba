#! /usr/bin/env python
from __future__ import print_function
from timeit import default_timer as time

import sys
import numpy as np
from numba import dppy
import dppy as ocldrv


@dppy.kernel(access_types={"read_only": ['a', 'b'], "write_only": ['c'], "read_write": []})
def data_parallel_sum(a, b, c):
    i = dppy.get_global_id(0)
    c[i] = a[i] + b[i]


global_size = 64
local_size = 32
N = global_size * local_size

a = np.array(np.random.random(N), dtype=np.float32)
b = np.array(np.random.random(N), dtype=np.float32)
c = np.ones_like(a)


def main():
    if ocldrv.has_gpu_device:
        with ocldrv.igpu_context(0) as device_env:
            print("----Running in GPU----")
            print("before A: ", a)
            print("before B: ", b)
            data_parallel_sum[global_size, local_size](a, b, c)
            print("after  C: ", c)
    elif ocldrv.has_cpu_device:
        with ocldrv.cpu_context(0) as device_env:
            print("----Running in CPU----")
            print("before A: ", a)
            print("before B: ", b)
            data_parallel_sum[global_size, local_size](a, b, c)
            print("after  C: ", c)
    else:
        print("No device found")
        exit()

    print("Done...")


if __name__ == '__main__':
    main()
