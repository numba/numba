#include "radixsort.cuh"

void compute_indices(
            uint8_t  *dev_data,
            unsigned *dev_indices,
            unsigned *dev_hist,
            unsigned *dev_bucket_total,
            unsigned ct_data,
            unsigned stride,
            unsigned offset,
            unsigned ct_block
)
{
    switch (stride) {
    case 4:
        cu_compute_indices_uint32<<<ct_block, 1>>>(
            (uint32_t*)dev_data,
            dev_indices,
            dev_hist,
            dev_bucket_total,
            ct_data,
            offset
        );
        break;
    default:
        cu_compute_indices_generic<<<ct_block, 1>>>(
            dev_data,
            dev_indices,
            dev_hist,
            dev_bucket_total,
            ct_data,
            stride,
            offset
        );
    }
    ASSERT_CUDA_LAST_ERROR();

}
