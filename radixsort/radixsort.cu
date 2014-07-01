#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#define CUDA_SAFE(X) {if((X) != cudaSuccess) {fprintf(stderr, "cuda error: line %d\n", __LINE__); exit(1);}}

enum { BITS = 8 };

__device__
unsigned global_id() {
    return threadIdx.x + blockIdx.x * blockDim.x;
}

__device__
unsigned local_id() {
    return threadIdx.x;
}


/**
blocksize >= BITS
*/
__global__
void
count_occurence(char input[], unsigned freq[], unsigned count)
{
    __shared__ unsigned cache[BITS];
    unsigned i = global_id();
    unsigned tid = local_id();

    if ( i >= count ) {
        return;
    }

    // Initialize cache
    if (tid < BITS) {
        cache[tid] = 0;
    }

    __syncthreads();

    // Compute local frequency
    atomicAdd(&cache[input[i]], 1);

    __syncthreads();

    // Update global frequency
    if (tid < BITS) {
        atomicAdd(&freq[tid], cache[tid]);
    }
}

/**
Launch BITS/2 blocksize
*/
__global__
void 
scan(unsigned freq[], unsigned offset[])
{
    __shared__ unsigned cache[BITS];
    unsigned tid = local_id();

    // Initialize cache
    if (tid < BITS) {
        if (tid == 0) cache[0] = 0;
        else {
            cache[tid] = freq[tid - 1];
        }
    }
    __syncthreads();

    unsigned round = 0;
    unsigned cond = BITS >> 1;
    while (cond) {
        unsigned step = 1 << round;
        if ( tid + step  < BITS ) {
            unsigned left = cache[tid];
            __syncthreads();

            cache[tid + step] += left;
            __syncthreads();
        }
        cond >>= 1;
        round += 1;
    }

    offset[tid] = cache[tid];
}

// __global__
// void 
// final_pos(char input[], unsigned offset[], unsigned indices[], unsigned count)
// {
//     unsigned bucket_offets[BITS] = {0};

//     for (unsigned i=0; i<count; ++i) {
//         unsigned off = bucket_offets[input[i]]++;
//         unsigned base = offset[input[i]];
//         indices[i] = base + off;
//     }
// }

// __global__
// void 
// scatter(char input[], unsigned indices[], char output[], unsigned count)
// {
//     for (unsigned i=0; i<count; ++i) {
//         output[indices[i]] = input[i];
//     }
// }

int main() {
    char data[] = {0, 2, 1, 4, 6, 5, 2, 4, 7};

    unsigned count = sizeof(data) / sizeof(char);
    unsigned data_size = count * sizeof(char);
    
    char output[count];

    unsigned freq[BITS];
    unsigned offset[BITS];
    unsigned indices[BITS];
    memset(freq, 0, sizeof(freq));

    unsigned freq_size = sizeof(freq);

    char *dev_data;
    unsigned *dev_freq, *dev_offset;

    CUDA_SAFE(cudaMalloc(&dev_data, data_size));
    CUDA_SAFE(cudaMalloc(&dev_freq, freq_size));
    CUDA_SAFE(cudaMalloc(&dev_offset, freq_size));

    CUDA_SAFE(cudaMemcpy(dev_data, data, data_size, cudaMemcpyHostToDevice));
    CUDA_SAFE(cudaMemcpy(dev_freq, freq, freq_size, cudaMemcpyHostToDevice));

    count_occurence<<<1, count>>>(dev_data, dev_freq, count);
    scan<<<1, BITS>>>(dev_freq, dev_offset);

    CUDA_SAFE(cudaMemcpy(freq, dev_freq, freq_size, cudaMemcpyDeviceToHost));
    CUDA_SAFE(cudaMemcpy(offset, dev_offset, freq_size, cudaMemcpyDeviceToHost));

    for(unsigned i=0; i<BITS; ++i) {
        printf("[%u] = %u\n", i, offset[i]);
    }
}
