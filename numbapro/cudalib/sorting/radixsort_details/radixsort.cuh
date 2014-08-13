#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <stdint.h>

#define CUDA_SAFE(X) {if((X) != cudaSuccess) {fprintf(stderr, "cuda error: line %d\n", __LINE__); exit(1);}}
#define ASSERT_CUDA_LAST_ERROR() CUDA_SAFE(cudaPeekAtLastError());

enum { BUCKET_SIZE = 256, BUCKET_MASK = 0xff };

extern "C" {

__global__
void cu_float_to_uint( uint32_t *in, uint32_t *out, unsigned count );

__global__
void cu_uint_to_float( uint32_t *in, uint32_t *out, unsigned count );


__global__
void cu_double_to_uint( uint64_t *in, uint64_t *out, unsigned count );

__global__
void cu_uint_to_double( uint64_t *in, uint64_t *out, unsigned count );

__global__
void cu_sign_fix_uint32( uint32_t *in, unsigned count );

__global__
void cu_sign_fix_uint64( uint64_t *in, unsigned count );

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
cu_compute_indices_uint64 (
    uint64_t  *data,
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

__global__
void cu_blockwise_sort_uint32(uint32_t *data,
                              unsigned *begin,
                              unsigned *count);

__global__
void cu_blockwise_sort_uint64(uint64_t *data,
                              unsigned *begin,
                              unsigned *count);

__global__
void cu_blockwise_argsort_uint32(uint32_t *data,
                                 unsigned *index,
                                 unsigned *begin,
                                 unsigned *count);

__global__
void cu_blockwise_argsort_uint64(uint64_t *data,
                                 unsigned *index,
                                 unsigned *begin,
                                 unsigned *count);


__global__
void cu_invert_uint32(uint32_t *data, unsigned count);

__global__
void cu_invert_uint64(uint64_t *data, unsigned count);

__global__
void cu_arange_uint32(uint32_t *data, unsigned count);

}; // end extern "C"

/* Reference
http://stereopsis.com/radix.html
*/

__global__
void cu_float_to_uint( uint32_t *in, uint32_t *out, unsigned count )
{
    unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= count ) return;

    uint32_t f = in[id];
    uint32_t mask = -int32_t(f >> 31) | 0x80000000ul;
	out[id] = f ^ mask;
}

__global__
void cu_uint_to_float( uint32_t *in, uint32_t *out, unsigned count )
{
    unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= count ) return;

    uint32_t f = in[id];
	uint32_t mask = ((f >> 31) - 1) | 0x80000000ul;
	out[id] = f ^ mask;
}

__global__
void cu_double_to_uint( uint64_t *in, uint64_t *out, unsigned count )
{
    unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= count ) return;

    uint64_t f = in[id];
    uint64_t mask =  -(f >> 63) | 0x8000000000000000ull ;
    out[id] = f ^ mask;
}

__global__
void cu_uint_to_double( uint64_t *in, uint64_t *out, unsigned count )
{
    unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= count ) return;

    uint64_t f = in[id];
	uint64_t mask = ((f >> 63) - 1) | 0x8000000000000000ull;
	out[id] = f ^ mask;
}


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
        enum { BITS = 8, BLOCK_SIZE=64, ILP=BUCKET_SIZE/BLOCK_SIZE };
        unsigned block_count = gridDim.x;
        unsigned blkid = blockIdx.x * BUCKET_SIZE;
        __shared__ unsigned occurences[BUCKET_SIZE];

        typedef cub::BlockLoad<unsigned*, BLOCK_SIZE, ILP,
                               cub::BLOCK_LOAD_VECTORIZE> LoadOcc;
        typedef cub::BlockStore<unsigned*, BLOCK_SIZE, ILP,
                                cub::BLOCK_STORE_VECTORIZE> StoreOcc;

        // Vector load causes load alignment error?
        // typedef cub::BlockLoad<Ty*, 1, ILP,
        //                        cub::BLOCK_LOAD_VECTORIZE> LoadData;

        typedef cub::BlockLoad<Ty*, 1, ILP> LoadData;
        typedef cub::BlockStore<unsigned*, 1, ILP,
                                cub::BLOCK_STORE_VECTORIZE> StoreIdx;

        __shared__ union {
            typename LoadOcc::TempStorage load_occ;
            typename StoreOcc::TempStorage store_occ;
            typename LoadData::TempStorage load_data;
            typename StoreIdx::TempStorage store_idx;
        } temp;

        union {
            unsigned occ[ILP];
            Ty data[ILP];
        } temp_threads;

        LoadOcc(temp.load_occ).Load(bucket_index, temp_threads.occ);
        __syncthreads();

        StoreOcc(temp.store_occ).Store(occurences, temp_threads.occ);
        __syncthreads();

        if (threadIdx.x != 0)
            return;
        // Block size is now 1

        // ILP optimized loop
        unsigned restart_from = 0;
        for (unsigned i=0; i<BUCKET_SIZE; i += ILP, restart_from = i) {
            unsigned id = blkid + i;
            if (id + ILP >= count) {
                break;
            }

            LoadData(temp.load_data).Load(&data[id], temp_threads.data);
            unsigned thread_idx[ILP];

            #pragma unroll 4
            for (unsigned j = 0; j < ILP; ++j) {
                uint8_t bucket = (temp_threads.data[j] >> (offset * BITS)) & BUCKET_MASK;
                unsigned histbase = hist[bucket * block_count + blockIdx.x];
                thread_idx[j] = histbase + occurences[bucket];
                occurences[bucket] += 1;
            }
            StoreIdx(temp.store_idx).Store(&indices[id], thread_idx);
        }

        // Fallback loop
        for (unsigned i=restart_from; i<BUCKET_SIZE; ++i) {
            unsigned id = blkid + i;
            if (id >= count) {
                return;
            }
            uint8_t bucket = (data[id] >> (offset * BITS)) & BUCKET_MASK;
            unsigned histbase = hist[bucket * block_count + blockIdx.x];
            unsigned idxoff = histbase + occurences[bucket];
            occurences[bucket] += 1;
            indices[id] = idxoff;
        }
    }
};

/*
This naive algorithm is faster than doing scan in the block or using
shared memory to do anything "smart" (at least for what I tried).

Must:
- gridsize == block count
- blocksize == 1
*/
__device__
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

// no export
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
cu_compute_indices_uint64 (
    uint64_t  *data,
    unsigned  *indices,
    unsigned *hist,
    unsigned *bucket_index,
    unsigned  count,
    unsigned  offset
)
{
    computer_indices<uint64_t>::kernel(
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
    if (id >= count)  {
        return;
    }
    unsigned dstid = indices[id];
    switch(stride){
    case 1:
        copy<uint8_t>::as(sorted, data, dstid, id);
        break;
    case 2:
        copy<uint16_t>::as(sorted, data, dstid, id);
        break;
    case 4:
        copy<uint32_t>::as(sorted, data, dstid, id);
        break;
    case 8:
        copy<uint64_t>::as(sorted, data, dstid, id);
        break;
    }
}

template<class T>
struct cu_blockwise_sort{
    __device__
    static void sort(T *data, unsigned *begin, unsigned *count, const T Maximum)
    {
        // Specialize for 1D block of 128 threads; 1 data per thread
        typedef cub::BlockRadixSort<T, 128, 1> BlockRadixSort;
        __shared__ typename BlockRadixSort::TempStorage temp_storage;
        unsigned blockoffset = begin[blockIdx.x];

        // Read
        unsigned localcount = count[blockIdx.x];
        const bool valid = threadIdx.x < localcount;

        if (localcount == 2) {
            if (threadIdx.x == 0) {
                T first = data[blockoffset + 0];
                T second = data[blockoffset + 1];
                if (first > second) {
                    data[blockoffset + 0] = second;
                    data[blockoffset + 1] = first;
                }
            }
        } else {
            T key[1];
            key[0] = Maximum;

            if (valid) {
                key[0] = data[blockoffset + threadIdx.x];
            }

            __syncthreads();

            // Sort
            BlockRadixSort(temp_storage).Sort(key);

            __syncthreads();

            // Write
            if (valid) {
                data[blockoffset + threadIdx.x] = key[0];
            }
        }
    }

    __device__
    static void argsort(T *data, unsigned *index, unsigned *begin,
                        unsigned *count, const T Maximum)
    {
        // Specialize for 1D block of 128 threads; 1 data per thread
        typedef cub::BlockRadixSort<T, 128, 1, unsigned> BlockRadixSort;
        __shared__ typename BlockRadixSort::TempStorage temp_storage;
        unsigned blockoffset = begin[blockIdx.x];

        // Read
        unsigned localcount = count[blockIdx.x];
        const bool valid = threadIdx.x < localcount;

        T key[1];
        key[0] = Maximum;

        unsigned value[1];
        value[0] = (unsigned)-1;

        if (valid) {
            key[0] = data[blockoffset + threadIdx.x];
            value[0] = index[blockoffset + threadIdx.x];
        }

        __syncthreads();

        // Sort
        BlockRadixSort(temp_storage).Sort(key, value);

        __syncthreads();

        // Write
        if (valid) {
            data[blockoffset + threadIdx.x] = key[0];
            index[blockoffset + threadIdx.x] = value[0];
        }
    }
};

__global__
void cu_blockwise_sort_uint32(uint32_t *data,
                              unsigned *begin,
                              unsigned *count)
{
    cu_blockwise_sort<uint32_t>::sort(data, begin, count, 0xffffffffu);
}

__global__
void cu_blockwise_sort_uint64(uint64_t *data,
                              unsigned *begin,
                              unsigned *count)
{
    cu_blockwise_sort<uint64_t>::sort(data, begin, count, 0xffffffffffffffffull);
}



__global__
void cu_blockwise_argsort_uint32(uint32_t *data,
                                 unsigned *index,
                                 unsigned *begin,
                                 unsigned *count)
{
    cu_blockwise_sort<uint32_t>::argsort(data, index, begin, count,
                                         0xffffffffu);
}

__global__
void cu_blockwise_argsort_uint64(uint64_t *data,
                                 unsigned *index,
                                 unsigned *begin,
                                 unsigned *count)

{
    cu_blockwise_sort<uint64_t>::argsort(data, index, begin, count,
                                         0xffffffffffffffffull);
}

template <class T>
struct signfix {
    __device__
    static void inplace(T &val) {
        T signbit = ((T)1) << (sizeof(T) * 8 - 1);
        val ^= signbit;
    }
};

__global__
void cu_sign_fix_uint32( uint32_t *in, unsigned count )
{
    unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= count ) return;

    signfix<uint32_t>::inplace(in[id]);
}

__global__
void cu_sign_fix_uint64( uint64_t *in, unsigned count )
{
    unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= count ) return;

    signfix<uint64_t>::inplace(in[id]);
}

template<class T>
struct inverter {
    __device__
    static void inplace(T & val) {
        val = -val;
    }
};

__global__
void cu_invert_uint32(uint32_t *data, unsigned count)
{
    unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= count ) return;

    inverter<uint32_t>::inplace(data[id]);
}

__global__
void cu_invert_uint64(uint64_t *data, unsigned count)
{
    unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= count ) return;

    inverter<uint64_t>::inplace(data[id]);
}

__global__
void cu_arange_uint32(uint32_t *data, unsigned count)
{
    unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= count ) return;
    data[id] = id;
}

