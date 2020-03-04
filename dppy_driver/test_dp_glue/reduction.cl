__kernel void reduceGPU (__global double *input,
                         const uint input_size,
                         __global double *partial_sums,
                         const uint partial_sum_size,
                         __local double *local_sums)
{
    uint local_id = get_local_id(0);
    uint global_id = get_global_id(0);
    uint group_size = get_local_size(0);

    // Copy from global to local memory
    local_sums[local_id] = 0;
    if (global_id < input_size) {
        local_sums[local_id] = input[global_id];
    }

    // Loop for computing localSums : divide WorkGroup into 2 parts
    for (uint stride = group_size/2; stride>0; stride >>=1) {
        // Waiting for each 2x2 addition into given workgroup
        barrier(CLK_LOCAL_MEM_FENCE);

        // Add elements 2 by 2 between local_id and local_id + stride
        if (local_id < stride) {
            local_sums[local_id] += local_sums[local_id + stride];
        }
    }

    if (local_id == 0) {
        partial_sums[get_group_id(0)] = local_sums[0];
        //printf("%d sum: %lf\n", get_group_id(0),  local_sums[0]);
    }
}
