from __future__ import print_function, division, absolute_import

import numpy as np
import math
import time

from numba import dppy
import dppy as ocldrv


@dppy.kernel
def reduction_kernel(A, R, stride):
    i = dppy.get_global_id(0)
    # sum two element
    R[i] = A[i] + A[i+stride]
    # store the sum to be used in nex iteration
    A[i] = R[i]


def test_sum_reduction():
    # Select a device for executing the kernel
    device_env = None

    try:
        device_env = ocldrv.runtime.get_cpu_device()
        print("Selected GPU device")
    except:
        try:
            device_env = ocldrv.runtime.get_cpu_device()
            print("Selected CPU device")
        except:
            print("No OpenCL devices found on the system")
            raise SystemExit()

    # This test will only work for even case
    N = 1024
    assert(N%2 == 0)

    A = np.array(np.random.random(N), dtype=np.float32)
    # at max we will require half the size of A to store sum
    R = np.array(np.random.random(math.ceil(N/2)), dtype=np.float32)

    if ocldrv.has_gpu_device:
        with ocldrv.igpu_context(0) as device_env:
            # create device array
            dA = device_env.copy_array_to_device(A)
            dR = device_env.copy_array_to_device(R)

            total = N

            while (total > 1):
                # call kernel
                global_size = total // 2
                reduction_kernel[global_size, dppy.DEFAULT_LOCAL_SIZE](dA, dR, global_size)
                total = total // 2

            device_env.copy_array_from_device(dR)
    else:
        print("No device found")
        exit()

    result = A.sum()
    max_abs_err = result - R[0]
    assert(max_abs_err < 1e-2)

if __name__ == '__main__':
    test_sum_reduction()
