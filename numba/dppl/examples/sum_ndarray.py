#! /usr/bin/env python
from __future__ import print_function
from timeit import default_timer as time

import sys
import numpy as np
from numba import dppl
import dpctl


@dppl.kernel(access_types={"read_only": ['a', 'b'], "write_only": ['c'], "read_write": []})
def data_parallel_sum(a, b, c):
    i = dppl.get_global_id(0)
    c[i] = a[i] + b[i]


global_size = 64
local_size = 32
N = global_size * local_size

a = np.array(np.random.random(N), dtype=np.float32)
b = np.array(np.random.random(N), dtype=np.float32)
c = np.ones_like(a)


def main():
    if dpctl.has_gpu_queues():
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            print("----Running in GPU----")
            print("before A: ", a)
            print("before B: ", b)
            data_parallel_sum[global_size, local_size](a, b, c)
            print("after  C: ", c)
    if dpctl.has_cpu_queues():
        with dpctl.device_context("opencl:cpu") as cpu_queue:
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
