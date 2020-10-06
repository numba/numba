from __future__ import print_function, division, absolute_import

import numpy as np
import math
import time

from numba import dppl
import dpctl


@dppl.kernel
def reduction_kernel(A, R, stride):
    i = dppl.get_global_id(0)
    # sum two element
    R[i] = A[i] + A[i+stride]
    # store the sum to be used in nex iteration
    A[i] = R[i]


def test_sum_reduction():
    # This test will only work for size = power of two
    N = 2048
    assert(N%2 == 0)

    A = np.array(np.random.random(N), dtype=np.float32)
    A_copy = A.copy()
    # at max we will require half the size of A to store sum
    R = np.array(np.random.random(math.ceil(N/2)), dtype=np.float32)

    if dpctl.has_gpu_queues():
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            total = N

            while (total > 1):
                # call kernel
                global_size = total // 2
                reduction_kernel[global_size, dppl.DEFAULT_LOCAL_SIZE](A, R, global_size)
                total = total // 2

    else:
        print("No device found")
        exit()

    result = A_copy.sum()
    max_abs_err = result - R[0]
    assert(max_abs_err < 1e-2)

if __name__ == '__main__':
    test_sum_reduction()
