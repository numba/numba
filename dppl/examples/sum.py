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
    c[i] = a[i] + b[i]


def driver(a, b, c, global_size):
    print("before : ", a)
    print("before : ", b)
    print("before : ", c)
    data_parallel_sum[global_size, dppl.DEFAULT_LOCAL_SIZE](a, b, c)
    print("after : ", c)


def main():
    global_size = 10
    N = global_size
    print("N", N)

    a = np.array(np.random.random(N), dtype=np.float32)
    b = np.array(np.random.random(N), dtype=np.float32)
    c = np.ones_like(a)

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
