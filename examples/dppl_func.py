import sys
import numpy as np
from numba import dppl
import math

import dpctl


@dppl.func
def g(a):
    return a + 1


@dppl.kernel
def f(a, b):
    i = dppl.get_global_id(0)
    b[i] = g(a[i])


def driver(a, b, N):
    print(b)
    print("--------")
    f[N, dppl.DEFAULT_LOCAL_SIZE](a, b)
    print(b)


def main():
    N = 10
    a = np.ones(N)
    b = np.ones(N)

    if dpctl.has_gpu_queues():
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            driver(a, b, N)
    elif dpctl.has_cpu_queues():
        with dpctl.device_context("opencl:cpu") as cpu_queue:
            driver(a, b, N)
    else:
        print("No device found")


if __name__ == '__main__':
    main()
