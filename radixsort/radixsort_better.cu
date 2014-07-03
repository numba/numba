/**
Divide into 256 element block.

For each block:
- build histogram

**/
// compile with: nvcc -arch=sm_30 radixsort_better.cu -I./cub

#include <iostream>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>

#define CUDA_SAFE(X) {if((X) != cudaSuccess) {fprintf(stderr, "cuda error: line %d\n", __LINE__); exit(1);}}
#define ASSERT_CUDA_LAST_ERROR() CUDA_SAFE(cudaPeekAtLastError());

enum { BUCKET_SIZE = 256, BUCKET_MASK = 0xff };

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


template<typename Ty>
struct copy {
    __device__
    static
    void as(void *output, void *input, unsigned dstidx, unsigned srcidx) {
        ((Ty*)output)[dstidx] = ((Ty*)input)[srcidx];
    }
};


/*
This is faster?!

Must:
- blocksize == 1
*/
__global__
void
cu_compute_indices_naive(
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

/**

Must:
- blocksize == BUCKET_SIZE
**/
__global__
void
cu_compute_indices(
    uint8_t  *data,
    unsigned *indices,
    unsigned *hist,
    unsigned *bucket_index,
    unsigned  count,
    unsigned  stride,
    unsigned  offset
)
{
    const unsigned block_count = gridDim.x;
    unsigned tid = threadIdx.x;
    unsigned blkid = blockIdx.x * BUCKET_SIZE;
    unsigned loc = blkid + tid;

    __shared__ unsigned temp_count;
    __shared__ unsigned temp_offset[BUCKET_SIZE];
    __shared__ bool temp_mark[BUCKET_SIZE];
    unsigned local_offset = 0;

    // Determine bucket for each data
    unsigned bucket = (unsigned)-1; // invalid bucket

    if (loc < count) {
        bucket = data[loc * stride + offset] & BUCKET_MASK;
    }

    //// Determine offset
    // For each bucket
    for (unsigned b = 0; b < BUCKET_SIZE; ++b) {
        // Mark if data is in bucket
        const bool mark = bucket == b;
        temp_count = 0;
        temp_mark[tid] = mark;

        __syncthreads();

        if (mark) {
            // Count marked
            atomicAdd(&temp_count, 1);
        }

        __syncthreads();

        if (tid == 0 && temp_count > 1) {
            unsigned ct = 0;
            for(unsigned i=0; i<blockDim.x; ++i) {
                if(temp_mark[i]) {
                    temp_offset[i] = ct;
                    ct += 1;
                }
            }
        }
        __syncthreads();

        if (mark) {
            if (temp_count > 1) {
                local_offset = temp_offset[tid];
            } else {
                local_offset = 0;
            }
        }

        __syncthreads();

    }

    if (loc < count) {
        unsigned bucketbase = bucket_index[bucket];
        unsigned histbase = hist[bucket * block_count + blockIdx.x];
        unsigned dst = histbase + bucketbase + local_offset;
        indices[loc] = dst;
    }
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


/*
This is faster?!

Must:
- blocksize == 1
*/
__global__
void
cu_scatter_histogram_naive(
    uint8_t  *data,
    uint8_t  *sorted,
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
        __syncthreads();
    }
}

/**

Must:
- blocksize == BUCKET_SIZE
**/
__global__
void
cu_scatter_histogram(
    uint8_t  *data,
    uint8_t  *sorted,
    unsigned *hist,
    unsigned *bucket_index,
    unsigned  count,
    unsigned  stride,
    unsigned  offset
)
{
    using cub::BlockScan;
    typedef BlockScan<uint16_t, BUCKET_SIZE> BlockScanT;
    __shared__ BlockScanT::TempStorage temp_scan;

    const unsigned block_count = gridDim.x;
    unsigned tid = threadIdx.x;
    unsigned blkid = blockIdx.x * BUCKET_SIZE;
    unsigned loc = blkid + tid;
    unsigned local_offset = 0;

    // Determine bucket for each data
    unsigned bucket = (unsigned)-1; // invalid bucket

    if (loc < count) {
        bucket = data[loc * stride + offset] & BUCKET_MASK;
    }

    //// Determine offset
    // For each bucket
    for (unsigned b = 0; b < BUCKET_SIZE; ++b) {
        // Mark if data is in bucket
        const uint16_t mark = bucket == b;

        // Prefix sum
        uint16_t offset;
        BlockScanT(temp_scan).ExclusiveSum(mark, offset);
        __syncthreads();

        if (mark) {
            // Assign offset for marked items
            local_offset = offset;
        }
    }

    __syncthreads();

    // Scatter
    if (loc < count) {
        unsigned bucketbase = bucket_index[bucket];
        unsigned histbase = hist[bucket * block_count + blockIdx.x];
        unsigned dst = histbase + bucketbase + local_offset;

        switch(stride){
        case 1:
            copy<uint8_t>::as(sorted, data, dst, loc);
            break;
        case 2:
            copy<uint16_t>::as(sorted, data, dst, loc);
            break;
        case 4:
            copy<uint32_t>::as(sorted, data, dst, loc);
            break;
        case 8:
            copy<uint64_t>::as(sorted, data, dst, loc);
            break;
        }
    }

}


int
main()
{
	using std::cout;
	using std::endl;

    typedef uint32_t data_type;

	const unsigned stride = sizeof(data_type);
	// const unsigned offset = 0;

	unsigned ct_data = 10000;
	unsigned sz_data = sizeof(data_type) * ct_data;

    const unsigned ct_block = (ct_data + (BUCKET_SIZE-1)) / BUCKET_SIZE;
    cout << "ct_block = " << ct_block << '\n';

	unsigned ct_hist = ct_block * BUCKET_SIZE;
	unsigned sz_hist = sizeof(unsigned) * ct_hist;

    unsigned ct_bucket_total = BUCKET_SIZE;
    unsigned sz_bucket_total = sizeof(unsigned) * ct_bucket_total;

    unsigned sz_indices = sizeof(unsigned)*ct_data;

	data_type *data = new data_type[ct_data];
	unsigned *hist = new unsigned[ct_hist];
    unsigned *bucket_total = new unsigned[ct_bucket_total];

	for (unsigned i=0; i<ct_data; ++i) {
		data[i] = ct_data - i - 1;
	}

    uint8_t *dev_data;
    uint8_t *dev_sorted;
    unsigned *dev_hist;
    unsigned *dev_bucket_total;
    unsigned *dev_indices;

    cudaMalloc(&dev_data, sz_data);
    cudaMalloc(&dev_indices, sz_data);
    cudaMalloc(&dev_sorted, sz_indices);
    cudaMalloc(&dev_hist, sz_hist);
    cudaMalloc(&dev_bucket_total, sz_bucket_total);

    // send data

	cudaMemcpy(dev_data, data, sz_data, cudaMemcpyHostToDevice);
	ASSERT_CUDA_LAST_ERROR();


    // compute

    for (unsigned offset=0; offset < 4; ++offset) {

    	cu_build_histogram<<<ct_block, BUCKET_SIZE>>>(
    		dev_data,
    		dev_hist,
    		stride,
    		offset,
    		ct_data
    	);
    	ASSERT_CUDA_LAST_ERROR();

        cu_scan_histogram<<<BUCKET_SIZE, SCAN_HISTOGRAM_BLOCK_SIZE>>>(
            dev_hist,
            dev_bucket_total,
            ct_block
        );
        ASSERT_CUDA_LAST_ERROR();


        cu_scan_bucket_index<<<1, BUCKET_SIZE>>>(dev_bucket_total);
        ASSERT_CUDA_LAST_ERROR();

        cu_compute_indices_naive<<<ct_block, 1>>>(
            dev_data,
            dev_indices,
            dev_hist,
            dev_bucket_total,
            ct_data,
            stride,
            offset
        );

        // cu_compute_indices<<<ct_block, BUCKET_SIZE>>>(
        //     dev_data,
        //     dev_indices,
        //     dev_hist,
        //     dev_bucket_total,
        //     ct_data,
        //     stride,
        //     offset
        // );

        cu_scatter<<<ct_block, BUCKET_SIZE>>>(
            dev_data,
            dev_sorted,
            dev_indices,
            ct_data,
            stride
        );

        // cu_scatter_histogram_naive<<<ct_block, 1>>>(
        //     dev_data,
        //     dev_sorted,
        //     dev_hist,
        //     dev_bucket_total,
        //     ct_data,
        //     stride,
        //     offset
        // );

        // cu_scatter_histogram<<<ct_block, BUCKET_SIZE>>>(
        //     dev_data,
        //     dev_sorted,
        //     dev_hist,
        //     dev_bucket_total,
        //     ct_data,
        //     stride,
        //     offset
        // );
        ASSERT_CUDA_LAST_ERROR();

        cudaMemcpy(dev_data, dev_sorted, sz_data, cudaMemcpyDeviceToDevice);
        ASSERT_CUDA_LAST_ERROR();
    }
    // write back

 //    cudaMemcpy(hist, dev_hist, sz_hist, cudaMemcpyDeviceToHost);
 //    ASSERT_CUDA_LAST_ERROR();


 //    cudaMemcpy(bucket_total, dev_bucket_total, sz_bucket_total,
 //               cudaMemcpyDeviceToHost);
	// ASSERT_CUDA_LAST_ERROR();

    cudaMemcpy(data, dev_sorted, sz_data, cudaMemcpyDeviceToHost);
    ASSERT_CUDA_LAST_ERROR();

    // cout << "hist\n";
    // for (unsigned i=0; i<BUCKET_SIZE; ++i) {
    //     for (unsigned j=0; j<ct_block; ++j) {
    //        cout << "bucket " << i << " block " << j
    //             << " = " << hist[ct_block * i + j] << '\n';
    //     }
    // }
    // cout << "bucket total\n";
    // for (unsigned i=0; i<ct_bucket_total; ++i) {
    //     cout << i << ' ' << bucket_total[i] << '\n';
    // }

    cout << "sorted\n";
    for (unsigned i = 0; i < ct_data; ++i) {
        // cout << i << ' ' << data[i] << '\n';
        if(data[i] != i) {
            cout << "error at i = " << i << " = " << data[i] << endl;
            exit(1);
        }
    }


	cout << "ok" << endl;
	return 0;
}