// compile with: nvcc -arch=sm_30 radixsort.cu -I./cub

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>

#define CUDA_SAFE(X) {if((X) != cudaSuccess) {fprintf(stderr, "cuda error: line %d\n", __LINE__); exit(1);}}
#define ASSERT_CUDA_LAST_ERROR() CUDA_SAFE(cudaPeekAtLastError());

enum { BUCKETSIZE = 256 };

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
gpu_count_occurrence(unsigned char input[], unsigned freq[], unsigned count,
                     unsigned stride, unsigned offset)
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
    atomicAdd(&cache[input[i * stride + offset]], 1);

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
gpu_scan(unsigned freq[])
{
    using namespace cub;
    enum { BLOCK_THREADS = BUCKETSIZE, ITEMS_PER_THREAD = 1 };
    typedef BlockLoad<unsigned*, BLOCK_THREADS, ITEMS_PER_THREAD> BlockLoadT;
    typedef BlockStore<unsigned*, BLOCK_THREADS, ITEMS_PER_THREAD> BlockStoreT;
    typedef BlockScan<unsigned, BLOCK_THREADS> BlockScanT;
    // Shared memory
    __shared__ union
    {
        typename BlockLoadT::TempStorage    load;
        typename BlockStoreT::TempStorage   store;
        typename BlockScanT::TempStorage    scan;
    } temp_storage;
    // Per-thread tile data
    unsigned data[ITEMS_PER_THREAD];
    // Load items into a blocked arrangement
    BlockLoadT(temp_storage.load).Load(freq, data);
    // Barrier for smem reuse
    __syncthreads();

    // Compute exclusive prefix sum
    unsigned aggregate;
    BlockScanT(temp_storage.scan).ExclusiveSum(data, data, aggregate);

    // Barrier for smem reuse
    __syncthreads();

    // Store items from a blocked arrangement
    BlockStoreT(temp_storage.store).Store(freq, data);

}


/*
Compute index base on bucket offset and input data.

blocksize >= BUCKETSIZE
*/
__global__
void
gpu_final_pos(unsigned char input[], unsigned starts[], unsigned indices[],
              unsigned count, unsigned stride, unsigned offset)
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
        cached_offset[tid] = starts[tid];
    }

    __syncthreads();

    // Scan over all data and assign the final position
    while (input_offset < count) {
        unsigned idx = input_offset + tid;
        if ( idx < count ) {
            unsigned char val = input[idx * stride + offset];
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

template<typename Ty>
struct copy {
    __device__
    static
    void as(void *output, void *input, unsigned *indices, unsigned i) {
        ((Ty*)output)[indices[i]] = ((Ty*)input)[i];
    }
};

__global__
void
gpu_scatter(void *input, unsigned indices[], void* output, unsigned count, unsigned bytesize)
{
    unsigned i = global_id();
    if ( i < count ) {
        switch(bytesize) {
        case 1:
            copy<uint8_t>::as(output, input, indices, i);
            break;
        case 2:
            copy<uint16_t>::as(output, input, indices, i);
            break;
        case 4:
            copy<uint32_t>::as(output, input, indices, i);
            break;
        case 8:
            copy<uint64_t>::as(output, input, indices, i);
            break;
        }

    }
}

unsigned
forall_gridsize_given_blocksize(unsigned count, unsigned blocksz) {
    return (count + (blocksz - 1)) / blocksz;
}

void
count_occurrence(unsigned char *dev_data, unsigned *dev_freq, unsigned count,
                 unsigned stride, unsigned offset) {
    unsigned blocksz = 512;
    unsigned gridsz = forall_gridsize_given_blocksize(count, blocksz);
    gpu_count_occurrence<<<gridsz, blocksz>>>(dev_data, dev_freq, count,
                                              stride, offset);
    ASSERT_CUDA_LAST_ERROR();
}

void
scan(unsigned *dev_freq) {
    gpu_scan<<<1, BUCKETSIZE>>>(dev_freq);
    ASSERT_CUDA_LAST_ERROR();
}

void
final_pos(unsigned char *dev_data, unsigned *dev_offset, unsigned *dev_indices,
          unsigned count, unsigned stride, unsigned offset) {
    gpu_final_pos<<<BUCKETSIZE, BUCKETSIZE>>>(dev_data, dev_offset, dev_indices, count,
                                       stride, offset);
    ASSERT_CUDA_LAST_ERROR();
}

void scatter(unsigned char *dev_data, unsigned *dev_indices, unsigned char *dev_sorted, unsigned count, unsigned bytesize) {
    unsigned blocksz = 512;
    unsigned gridsz = forall_gridsize_given_blocksize(count, blocksz);
    gpu_scatter<<<gridsz, blocksz>>>(dev_data, dev_indices, dev_sorted, count, bytesize);
    ASSERT_CUDA_LAST_ERROR();
}

void
radixsort(unsigned char *dev_data,
          unsigned      *dev_freq,
          unsigned      *dev_indices,
          unsigned char *dev_sorted,
          unsigned       count,
          unsigned       elemsize,
          unsigned       offset)
{
    cudaMemset(dev_freq, 0, sizeof(unsigned)*BUCKETSIZE);
    count_occurrence(dev_data, dev_freq, count, elemsize, offset);
    scan(dev_freq);
    final_pos(dev_data, dev_freq, dev_indices, count, elemsize, offset);
    scatter(dev_data, dev_indices, dev_sorted, count, elemsize);
}


enum {COUNT = 300};

int main() {
    typedef int data_type;
    data_type data[COUNT] = {0};
    const unsigned elemsize = sizeof(data_type);
    const unsigned offset = 0;

    const unsigned count = sizeof(data) / sizeof(data_type);

    for(unsigned i=0; i<count; ++i) {
        data[i] = count - i - 1;
    }

    unsigned data_size = sizeof(data);

    unsigned freq[BUCKETSIZE];
    unsigned indices[count];
    memset(freq, 0, sizeof(freq));
    memset(indices, 0, sizeof(indices));

    unsigned freq_size = sizeof(freq);
    unsigned indices_size = sizeof(indices);

    unsigned char *dev_data, *dev_sorted;
    unsigned *dev_freq, *dev_indices;

    CUDA_SAFE(cudaMalloc(&dev_data, data_size));
    CUDA_SAFE(cudaMalloc(&dev_freq, freq_size));
    CUDA_SAFE(cudaMalloc(&dev_indices, indices_size));
    CUDA_SAFE(cudaMalloc(&dev_sorted, data_size));

    CUDA_SAFE(cudaMemcpy(dev_data, data, data_size, cudaMemcpyHostToDevice));
    //CUDA_SAFE(cudaMemcpy(dev_freq, freq, freq_size, cudaMemcpyHostToDevice));

    if (0) {

        // MSD sort
        radixsort(dev_data, dev_freq, dev_indices, dev_sorted, count, elemsize,
                  1);

        CUDA_SAFE(cudaMemcpy(dev_data, dev_sorted, data_size, cudaMemcpyDeviceToDevice));
        radixsort(dev_data, dev_freq, dev_indices, dev_sorted, 256, elemsize,
                  0);

        CUDA_SAFE(cudaMemcpy(dev_data, dev_sorted, data_size, cudaMemcpyDeviceToDevice));
        radixsort(dev_data + 4*256, dev_freq, dev_indices, dev_sorted + 4*256, 300 - 256, elemsize,
                  0);
    } else {
        // LSD sort
        radixsort(dev_data, dev_freq, dev_indices, dev_sorted, count, elemsize, 0);
        CUDA_SAFE(cudaMemcpy(dev_data, dev_sorted, data_size, cudaMemcpyDeviceToDevice));
        radixsort(dev_data, dev_freq, dev_indices, dev_sorted, count, elemsize, 1);
    }

    CUDA_SAFE(cudaMemcpy(freq, dev_freq, freq_size, cudaMemcpyDeviceToHost));
    // CUDA_SAFE(cudaMemcpy(indices, dev_indices, indices_size, cudaMemcpyDeviceToHost));
    CUDA_SAFE(cudaMemcpy(data, dev_sorted, data_size, cudaMemcpyDeviceToHost));

    puts("offset");
    for(unsigned i=0; i<BUCKETSIZE; ++i) {
        printf("[%u] = %u\n", i, freq[i]);
    }
    // puts("final pos");
    // for(unsigned i=0; i<count; ++i) {
    //     printf("[%u] = %u\n", i, indices[i]);
    // }
    puts("sorted");
    for(unsigned i=0; i<count; ++i) {
        printf("[%u] = %u\n", i, data[i]);
    }
}
