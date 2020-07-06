import sys
import numpy as np
from numba import dppy, int32
import math

import dppy as ocldrv


def recursive_reduction(size, group_size,
                        Dinp, Dpartial_sums, device_env):

    @dppy.kernel
    def sum_reduction_kernel(inp, input_size,
                             partial_sums):
        local_id   = dppy.get_local_id(0)
        global_id  = dppy.get_global_id(0)
        group_size = dppy.get_local_size(0)
        group_id   = dppy.get_group_id(0)

        local_sums = dppy.local.static_alloc(64, int32)

        local_sums[local_id] = 0

        if global_id < input_size:
            local_sums[local_id] = inp[global_id]

        # Loop for computing local_sums : divide workgroup into 2 parts
        stride = group_size // 2
        while (stride > 0):
            # Waiting for each 2x2 addition into given workgroup
            dppy.barrier(dppy.CLK_LOCAL_MEM_FENCE)

            # Add elements 2 by 2 between local_id and local_id + stride
            if (local_id < stride):
                local_sums[local_id] += local_sums[local_id + stride]

            stride >>= 1

        if local_id == 0:
            partial_sums[group_id] = local_sums[0]


    result = 0
    nb_work_groups = 0
    passed_size = size

    if (size <= group_size):
        nb_work_groups = 1
    else:
        nb_work_groups = size // group_size;
        if (size % group_size != 0):
            nb_work_groups += 1
            passed_size = nb_work_groups * group_size

    sum_reduction_kernel[passed_size, group_size](Dinp, size, Dpartial_sums)

    if nb_work_groups <= group_size:
        sum_reduction_kernel[group_size, group_size](Dpartial_sums, nb_work_groups, Dinp)
        device_env.copy_array_from_device(Dinp)
        result = Dinp._ndarray[0]
    else:
        result = recursive_reduction(nb_work_groups, group_size,
                                     Dpartial_sums, Dinp, device_env)

    return result


def sum_reduction_recursive():
    global_size = 20000
    work_group_size = 64
    nb_work_groups = global_size // work_group_size
    if (global_size % work_group_size) != 0:
        nb_work_groups += 1

    inp = np.ones(global_size).astype(np.int32)
    partial_sums = np.zeros(nb_work_groups).astype(np.int32)


    if ocldrv.has_gpu_device:
        with ocldrv.igpu_context(0) as device_env:
            Dinp = device_env.copy_array_to_device(inp)
            Dpartial_sums = device_env.copy_array_to_device(partial_sums)

            print("Running recursive reduction")
            result = recursive_reduction(global_size, work_group_size,
                                         Dinp, Dpartial_sums, device_env)
    else:
        print("No device found")
        exit()


    print("Expected:", global_size, "--- GOT:", result)
    assert(result == global_size)


sum_reduction_recursive()
