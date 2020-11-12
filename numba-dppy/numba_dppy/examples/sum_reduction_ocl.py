import sys
import numpy as np
from numba import dppl, int32
import math

import dpctl

def sum_reduction_device_plus_host():
    @dppl.kernel
    def sum_reduction_kernel(inp, partial_sums):
        local_id   = dppl.get_local_id(0)
        global_id  = dppl.get_global_id(0)
        group_size = dppl.get_local_size(0)
        group_id   = dppl.get_group_id(0)

        local_sums = dppl.local.static_alloc(64, int32)

        # Copy from global to local memory
        local_sums[local_id] = inp[global_id]

        # Loop for computing local_sums : divide workgroup into 2 parts
        stride = group_size // 2
        while (stride > 0):
            # Waiting for each 2x2 addition into given workgroup
            dppl.barrier(dppl.CLK_LOCAL_MEM_FENCE)

            # Add elements 2 by 2 between local_id and local_id + stride
            if (local_id < stride):
                local_sums[local_id] += local_sums[local_id + stride]

            stride >>= 1

        if local_id == 0:
            partial_sums[group_id] = local_sums[0]

    global_size = 1024
    work_group_size = 64
    # nb_work_groups have to be even for this implementation
    nb_work_groups = global_size // work_group_size

    inp = np.ones(global_size).astype(np.int32)
    partial_sums = np.zeros(nb_work_groups).astype(np.int32)

    if dpctl.has_gpu_queues():
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            print("Running Device + Host reduction")
            sum_reduction_kernel[global_size, work_group_size](inp, partial_sums)
    else:
        print("No device found")
        exit()

    final_sum = 0
    # calculate the final sum in HOST
    for i in range(nb_work_groups):
        final_sum += partial_sums[i]

    assert(final_sum == global_size)
    print("Expected:", global_size, "--- GOT:", final_sum)


if __name__ == '__main__':
    sum_reduction_device_plus_host()
