#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#define CUDA_SAFE(X) {if((X) != cudaSuccess) {fprintf(stderr, "cuda error: line %d\n", __LINE__); exit(1);}}
#define ASSERT_CUDA_LAST_ERROR() CUDA_SAFE(cudaPeekAtLastError());

enum { BUCKETSIZE = 16 };

__device__
unsigned global_id() {
    return threadIdx.x + blockIdx.x * blockDim.x;
}

__device__
unsigned local_id() {
    return threadIdx.x;
}


/**
blocksize >= BUCKETSIZE
*/
__global__
void
gpu_count_occurence(unsigned char input[], unsigned freq[], unsigned count)
{
    __shared__ unsigned cache[BUCKETSIZE];
    unsigned i = global_id();
    unsigned tid = local_id();

    if ( i >= count ) {
        return;
    }

    // Initialize cache
    if (tid < BUCKETSIZE) {
        cache[tid] = 0;
    }

    __syncthreads();

    // Compute local frequency
    atomicAdd(&cache[input[i]], 1);

    __syncthreads();

    // Update global frequency
    if (tid < BUCKETSIZE) {
        atomicAdd(&freq[tid], cache[tid]);
    }
}

/**
Launch BUCKETSIZE blocksize
*/
__global__
void 
gpu_scan(unsigned freq[], unsigned offset[])
{
    __shared__ unsigned cache[BUCKETSIZE];
    unsigned tid = local_id();

    // Initialize cache
    if (tid < BUCKETSIZE) {
        if (tid == 0) cache[0] = 0;
        else {
            cache[tid] = freq[tid - 1];
        }
    }
    __syncthreads();

    // Perform scan on cache
    unsigned round = 0;
    unsigned cond = BUCKETSIZE >> 1;
    while (cond) {
        unsigned step = 1 << round;
        if ( tid + step  < BUCKETSIZE ) {
            unsigned left = cache[tid];
            __syncthreads();

            cache[tid + step] += left;
            __syncthreads();
        }
        cond >>= 1;
        round += 1;
    }

    // Copy result in cache to global memory
    offset[tid] = cache[tid];
}


/*
Compute index base on bucket offset and input data.
*/
__global__
void 
gpu_final_pos(unsigned char input[], unsigned offset[], unsigned indices[], unsigned count)
{
    __shared__ unsigned bucket_offset;
    __shared__ unsigned cached_offset[BUCKETSIZE];
    unsigned bucket_idx = blockIdx.x;
    unsigned tid = threadIdx.x;
    unsigned input_offset = 0;

    // Preload
    if (tid == 0) {
        // First thread initialize bucket_offset
        bucket_offset = 0; 
    }

    if (tid < BUCKETSIZE) {
        // Cache offset memory into shared memory
        cached_offset[tid] = offset[tid];
    }

    __syncthreads();

    // Scan over all data and assign the final position
    while (input_offset < count) {
        unsigned idx = input_offset + tid;
        if ( idx < count ) {
            unsigned char val = input[idx];    
            if (val == bucket_idx) {    // is in this bucket
                unsigned base = cached_offset[val];
                unsigned off = atomicAdd(&bucket_offset, 1);
                indices[idx] = base + off;
            }
        }
        // Advance by block size
        input_offset += blockDim.x;
        // Barrier
        __syncthreads();
    }
}

__global__
void 
gpu_scatter(unsigned char input[], unsigned indices[], unsigned char output[], unsigned count)
{
    unsigned i = global_id();
    if ( i < count ) {
        output[indices[i]] = input[i];
    }
}

enum {COUNT = 1024};

unsigned 
forall_gridsize_given_blocksize(unsigned count, unsigned blocksz) {
    return (count + (blocksz - 1)) / blocksz;
}

void
count_occurence(unsigned char *dev_data, unsigned *dev_freq, unsigned count) {
    unsigned blocksz = 512;
    unsigned gridsz = forall_gridsize_given_blocksize(count, blocksz);
    gpu_count_occurence<<<gridsz, blocksz>>>(dev_data, dev_freq, count);
    ASSERT_CUDA_LAST_ERROR();
}

void 
scan(unsigned *dev_freq, unsigned *dev_offset) {
    gpu_scan<<<1, BUCKETSIZE>>>(dev_freq, dev_offset);
    ASSERT_CUDA_LAST_ERROR();   
}

void 
final_pos(unsigned char *dev_data, unsigned *dev_offset, unsigned *dev_indices, unsigned count) {
    gpu_final_pos<<<BUCKETSIZE, 512>>>(dev_data, dev_offset, dev_indices, count);
    ASSERT_CUDA_LAST_ERROR();
}   

void scatter(unsigned char *dev_data, unsigned *dev_indices, unsigned char *dev_sorted, unsigned count) {
    unsigned blocksz = 512;
    unsigned gridsz = forall_gridsize_given_blocksize(count, blocksz);
    gpu_scatter<<<gridsz, blocksz>>>(dev_data, dev_indices, dev_sorted, count);
    ASSERT_CUDA_LAST_ERROR();
}

int main() {
    unsigned char data[COUNT] = {0};
    const unsigned count = sizeof(data) / sizeof(unsigned char);

    for(unsigned i=0; i<count; ++i) {
        data[i] = (i % BUCKETSIZE);
    }

    unsigned data_size = sizeof(data);

    unsigned freq[BUCKETSIZE];
    unsigned offset[BUCKETSIZE];
    unsigned indices[count];
    memset(freq, 0, sizeof(freq));
    memset(indices, 0, sizeof(indices));

    unsigned freq_size = sizeof(freq);
    unsigned indices_size = sizeof(indices);

    unsigned char *dev_data, *dev_sorted;
    unsigned *dev_freq, *dev_offset, *dev_indices;

    CUDA_SAFE(cudaMalloc(&dev_data, data_size));
    CUDA_SAFE(cudaMalloc(&dev_freq, freq_size));
    CUDA_SAFE(cudaMalloc(&dev_offset, freq_size));
    CUDA_SAFE(cudaMalloc(&dev_indices, indices_size));
    CUDA_SAFE(cudaMalloc(&dev_sorted, data_size));

    CUDA_SAFE(cudaMemcpy(dev_data, data, data_size, cudaMemcpyHostToDevice));
    CUDA_SAFE(cudaMemcpy(dev_freq, freq, freq_size, cudaMemcpyHostToDevice));

    count_occurence(dev_data, dev_freq, count);
    scan(dev_freq, dev_offset);
    final_pos(dev_data, dev_offset, dev_indices, count);
    scatter(dev_data, dev_indices, dev_sorted, count);

    // CUDA_SAFE(cudaMemcpy(freq, dev_freq, freq_size, cudaMemcpyDeviceToHost));
    CUDA_SAFE(cudaMemcpy(offset, dev_offset, freq_size, cudaMemcpyDeviceToHost));
    CUDA_SAFE(cudaMemcpy(indices, dev_indices, indices_size, cudaMemcpyDeviceToHost));
    CUDA_SAFE(cudaMemcpy(data, dev_sorted, data_size, cudaMemcpyDeviceToHost));

    puts("offset");
    for(unsigned i=0; i<BUCKETSIZE; ++i) {
        printf("[%u] = %u\n", i, offset[i]);
    }
    puts("final pos");
    for(unsigned i=0; i<count; ++i) {
        printf("[%u] = %u\n", i, indices[i]);
    }
    puts("sorted");
    for(unsigned i=0; i<count; ++i) {
        printf("[%u] = %u\n", i, data[i]);
    }
}
