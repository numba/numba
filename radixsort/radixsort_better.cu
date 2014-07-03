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
        __syncthreads();
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

Must:
- blocksize == BUCKET_SIZE
*/
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

	unsigned ct_data = 1024 * 1024 * 10;
	unsigned sz_data = sizeof(data_type) * ct_data;

    unsigned ct_block = (ct_data + (BUCKET_SIZE-1)) / BUCKET_SIZE;

	unsigned ct_hist = ct_block * BUCKET_SIZE;
	unsigned sz_hist = sizeof(unsigned) * ct_hist;

    unsigned ct_bucket_total = BUCKET_SIZE;
    unsigned sz_bucket_total = sizeof(unsigned) * ct_bucket_total;

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

	cudaMalloc(&dev_data, sz_data);
    cudaMalloc(&dev_sorted, sz_data);
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

        cu_scatter_histogram<<<ct_block, 1>>>(
            dev_data,
            dev_sorted,
            dev_hist,
            dev_bucket_total,
            ct_data,
            stride,
            offset
        );
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