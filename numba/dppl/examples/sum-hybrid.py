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


N = 50*32
global_size = N,


def main():
    if dpctl.has_cpu_queues():
        with dpctl.device_context("opencl:cpu") as cpu_queue:
            print("-----Running in CPU-----")
            a = np.array(np.random.random(N), dtype=np.float32)
            b = np.array(np.random.random(N), dtype=np.float32)
            c = np.ones_like(a)
            print("before A: ", a)
            print("before B: ", b)
            data_parallel_sum[global_size, dppl.DEFAULT_LOCAL_SIZE](a, b, c)
            print("after  C: ", c)
    else:
        print("CPU device not found")

    if dpctl.has_gpu_queues():
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            print("-----Running in GPU-----")
            a = np.array(np.random.random(N), dtype=np.float32)
            b = np.array(np.random.random(N), dtype=np.float32)
            c = np.ones_like(a)
            print("before A: ", a)
            print("before B: ", b)
            data_parallel_sum[global_size, dppl.DEFAULT_LOCAL_SIZE](a, b, c)
            print("after  C: ", c)
    else:
        print("GPU device not found")
        exit()

    print("Done...")


if __name__ == '__main__':
    main()
