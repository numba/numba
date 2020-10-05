#! /usr/bin/env python
from __future__ import print_function
from timeit import default_timer as time

import sys
import numpy as np
from numba import dppl
import dpctl


@dppl.kernel
def data_parallel_sum(a, b, c):
    i = dppl.get_global_id(0)
    j = dppl.get_global_id(1)
    c[i,j] = a[i,j] + b[i,j]


def driver(a, b, c, global_size):
    print("before A: ", a)
    print("before B: ", b)
    data_parallel_sum[global_size, dppl.DEFAULT_LOCAL_SIZE](a, b, c)
    print("after  C : ", c)


def main():
    # Array dimesnions
    X = 8
    Y = 8
    global_size = X,Y

    a = np.arange(X*Y, dtype=np.float32).reshape(X,Y)
    b = np.array(np.random.random(X*Y), dtype=np.float32).reshape(X,Y)
    c = np.ones_like(a).reshape(X,Y)

    if dpctl.has_gpu_queues():
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            driver(a, b, c, global_size)
    elif dpctl.has_cpu_queues():
        with dpctl.device_context("opencl:cpu") as cpu_queue:
            driver(a, b, c, global_size)
    else:
        print("No device found")
        exit()

    print("Done...")


if __name__ == '__main__':
    main()
