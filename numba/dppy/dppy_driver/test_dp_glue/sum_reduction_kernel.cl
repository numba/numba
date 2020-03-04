__kernel void sumGPU ( __global const double *input,
                       __global double *partialSums,
                       __local double *localSums,
                       __global double *finalSum)
 {
  uint local_id = get_local_id(0);
  uint global_size = get_global_size(0);
  uint group_size = get_local_size(0);
  uint global_id = get_global_id(0);

  // Copy from global to local memory
  localSums[local_id] = input[global_id];

  // Deal with odd case
  if (local_id == 0) {
    if ((group_size % 2) != 0)
    {
    localSums[local_id] += input[global_id + (group_size-1)];
    }
  }

  // Loop for computing localSums : divide WorkGroup into 2 parts
  for (uint stride = group_size/2; stride>0; stride >>=1)
     {
      // Waiting for each 2x2 addition into given workgroup
      barrier(CLK_LOCAL_MEM_FENCE);

      // Add elements 2 by 2 between local_id and local_id + stride
      if (local_id < stride)
        localSums[local_id] += localSums[local_id + stride];
     }

  // Write result into partialSums[nWorkGroups]
  if (local_id == 0)
    {
    partialSums[get_group_id(0)] = localSums[0];
    }

  barrier(CLK_GLOBAL_MEM_FENCE);
  // last sum
  if (global_id == 0) {
    uint idx;
    finalSum[0] = 0;
    for (idx = 0; idx < global_size/group_size; ++idx) {
      finalSum[0] += partialSums[idx];
    }
  }
 }


__kernel void sumGPUCPU ( __global const double *input,
                          __global double *partialSums,
                          __local double *localSums)
 {
  uint local_id = get_local_id(0);
  uint global_size = get_global_size(0);
  uint group_size = get_local_size(0);
  uint global_id = get_global_id(0);

  // Copy from global to local memory
  localSums[local_id] = input[global_id];

  // Deal with odd case
  if (local_id == 0) {
    if ((group_size % 2) != 0)
    {
    localSums[local_id] += input[global_id + (group_size-1)];
    }
  }

  // Loop for computing localSums : divide WorkGroup into 2 parts
  for (uint stride = group_size/2; stride>0; stride >>=1)
     {
      // Waiting for each 2x2 addition into given workgroup
      barrier(CLK_LOCAL_MEM_FENCE);

      // Add elements 2 by 2 between local_id and local_id + stride
      if (local_id < stride)
        localSums[local_id] += localSums[local_id + stride];
     }

  // Write result into partialSums[nWorkGroups]
  if (local_id == 0)
    {
    partialSums[get_group_id(0)] = localSums[0];
    }
 }

__kernel void scalingGPU ( __global double *input,
                           __global double *finalSum)
 {
  uint idx = get_global_id(0);

  input[idx] = input[idx] * finalSum[0];

 }

__kernel void scalingGPUCPU ( __global double *input,
                              const double scaling_factor)
 {
  uint idx = get_global_id(0);

  input[idx] = input[idx] * scaling_factor;

 }
