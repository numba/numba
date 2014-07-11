#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>

#define CUDA_SAFE(X) {if((X) != cudaSuccess) {fprintf(stderr, "cuda error: line %d\n", __LINE__); exit(1);}}
#define ASSERT_CUDA_LAST_ERROR() CUDA_SAFE(cudaPeekAtLastError());

enum { BUCKET_SIZE = 256, BUCKET_MASK = 0xff };

extern "C" {
__global__
void cu_build_histogram(
    uint8_t  *data,
    unsigned *hist,
    unsigned  stride,
    unsigned  offset,
    unsigned  count
);

__global__
void
cu_scan_histogram(
    unsigned *hist,
    unsigned *bucket_total,
    unsigned blockcount
);

__global__
void
cu_compute_indices_uint32 (
    uint32_t  *data,
    unsigned  *indices,
    unsigned *hist,
    unsigned *bucket_index,
    unsigned  count,
    unsigned  offset
);

__global__
void
cu_scan_bucket_index(unsigned *bucket_index);

__global__
void
cu_scatter(
    void     *data,
    void     *sorted,
    unsigned *indices,
    unsigned  count,
    unsigned  stride
);

// end extern "C"
};

/*
Must
- blocksize == BUCKET_SIZE
*/
__global__
void cu_build_histogram(
	uint8_t  *data,
	unsigned *hist,
	unsigned  stride,
	unsigned  offset,
	unsigned  count
)
{
	__shared__ unsigned sm_counter[BUCKET_SIZE];

	unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned tid = threadIdx.x;

    // Initialize counter

	sm_counter[tid] = 0;

	__syncthreads();

    // Binning
    if (id < count) {
	   uint8_t bval = data[id * stride + offset] & BUCKET_MASK;
	   atomicAdd(&sm_counter[bval], 1);
    }

	__syncthreads();

    // Store in a transposed manner such that
    // blocks of the same bucket is in a row
   hist[tid * gridDim.x + blockIdx.x] = sm_counter[tid];
}


enum {
   SCAN_HISTOGRAM_BLOCK_SIZE = 256
};

/**
Must
- gridsize == bucket_size
- blocksize == SCAN_BLOCK_SUM_BLOCK_SIZE
**/
__global__
void
cu_scan_histogram(
    unsigned *hist,
    unsigned *bucket_total,
    unsigned blockcount
)
{
    using cub::BlockScan;
    typedef BlockScan<unsigned, SCAN_HISTOGRAM_BLOCK_SIZE> BlockScanT;
    __shared__ BlockScanT::TempStorage temp_scan;

    unsigned total = 0;
    for (unsigned loc_base = 0; loc_base < blockcount;
        loc_base += SCAN_HISTOGRAM_BLOCK_SIZE)
    {
        unsigned block_offset = threadIdx.x + loc_base;
        unsigned loc = blockIdx.x * blockcount + block_offset;

        unsigned data = 0;
        unsigned aggregate;
        if (block_offset < blockcount) {
            data = hist[loc];
        }

        BlockScanT(temp_scan).ExclusiveSum(data, data, aggregate);
        __syncthreads();

        if (block_offset < blockcount) {
            hist[loc] = total + data;
        }
        total += aggregate;
    }

    if (threadIdx.x == 0) {
        bucket_total[blockIdx.x] = total;
    }
}

/*

Must:
- blocksize == BUCKET_SIZE
*/
__global__
void
cu_scan_bucket_index(unsigned *bucket_index)
{
    using cub::BlockScan;
    typedef BlockScan<unsigned, BUCKET_SIZE> BlockScanT;
    __shared__ BlockScanT::TempStorage temp_scan;
    unsigned data = bucket_index[threadIdx.x];
    BlockScanT(temp_scan).ExclusiveSum(data, data);
    __syncthreads();
    bucket_index[threadIdx.x] = data;
}

namespace {
template<typename Ty>
struct copy {
    __device__
    static
    void as(void *output, void *input, unsigned dstidx, unsigned srcidx) {
        ((Ty*)output)[dstidx] = ((Ty*)input)[srcidx];
    }
};


template<typename Ty>
struct computer_indices {
    __device__
    static void kernel(
        Ty  *data,
        unsigned  *indices,
        unsigned *hist,
        unsigned *bucket_index,
        unsigned  count,
        unsigned  offset
    )
    {
        enum { BITS = 8 };
        unsigned block_count = gridDim.x;
        unsigned blkid = blockIdx.x * BUCKET_SIZE;
        unsigned occurences[BUCKET_SIZE];

        for (unsigned i=0; i<BUCKET_SIZE; ++i) {
            occurences[i] = 0;
        }

        for (unsigned i=0; i<BUCKET_SIZE; ++i) {
            unsigned id = blkid + i;
            if (id >= count) {
                return;
            }
            uint8_t bucket = (data[id] >> (offset * BITS)) & BUCKET_MASK;
            unsigned bucketbase = bucket_index[bucket];
            unsigned histbase = hist[bucket * block_count + blockIdx.x];
            unsigned offset = histbase + bucketbase + occurences[bucket];
            occurences[bucket] += 1;

            indices[id] = offset;
        }
    }
};

// no export
}

/*
This naive algorithm is faster than doing scan in the block or using
shared memory to do anything "smart" (at least for what I tried).

Must:
- gridsize == block count
- blocksize == 1
*/
__global__
void
cu_compute_indices_generic(
    uint8_t  *data,
    unsigned  *indices,
    unsigned *hist,
    unsigned *bucket_index,
    unsigned  count,
    unsigned  stride,
    unsigned  offset
)
{
    unsigned block_count = gridDim.x;
    unsigned blkid = blockIdx.x * BUCKET_SIZE;
    unsigned occurences[BUCKET_SIZE];

    for (unsigned i=0; i<BUCKET_SIZE; ++i) {
        occurences[i] = 0;
    }

    for (unsigned i=0; i<BUCKET_SIZE; ++i) {
        unsigned id = blkid + i;
        if (id >= count) {
            return;
        }
        uint8_t bucket = data[id * stride + offset] & BUCKET_MASK;
        unsigned bucketbase = bucket_index[bucket];
        unsigned histbase = hist[bucket * block_count + blockIdx.x];
        unsigned offset = histbase + bucketbase + occurences[bucket];
        occurences[bucket] += 1;

        indices[id] = offset;
    }
}

__global__
void
cu_compute_indices_uint32 (
    uint32_t  *data,
    unsigned  *indices,
    unsigned *hist,
    unsigned *bucket_index,
    unsigned  count,
    unsigned  offset
)
{
    computer_indices<uint32_t>::kernel(
        data, indices, hist, bucket_index, count, offset);
}

__global__
void
cu_scatter(
    void     *data,
    void     *sorted,
    unsigned *indices,
    unsigned  count,
    unsigned  stride
)
{
    unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned offset = indices[id];
    if (id >= count)  {
        return;
    }
    switch(stride){
    case 1:
        copy<uint8_t>::as(sorted, data, offset, id);
        break;
    case 2:
        copy<uint16_t>::as(sorted, data, offset, id);
        break;
    case 4:
        copy<uint32_t>::as(sorted, data, offset, id);
        break;
    case 8:
        copy<uint64_t>::as(sorted, data, offset, id);
        break;
    }
}

