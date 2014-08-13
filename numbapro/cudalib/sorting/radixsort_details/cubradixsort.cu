#include <cub/device/device_radix_sort.cuh>
#include <stdint.h>

template <class Tk, class Tv=unsigned>
struct RadixSort {
    static
    void sort(  unsigned  num_items,
                Tk  *d_key_buf,
                Tk  *d_key_alt_buf,
                Tv  *d_value_buf,
                Tv  *d_value_alt_buf,
                cudaStream_t stream,
                int descending,
                unsigned begin_bit,
                unsigned end_bit      )
    {

        void     *d_temp_storage = NULL;
        size_t   temp_storage_bytes = 0;
        cub::DoubleBuffer<Tk> d_keys(d_key_buf, d_key_alt_buf);
        if (d_value_buf) {
            cub::DoubleBuffer<Tv> d_values(d_value_buf, d_value_alt_buf);
            if (descending) {
                cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, stream);

                cudaMalloc(&d_temp_storage, temp_storage_bytes);
                cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, stream);
            } else {
                cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, stream);

                cudaMalloc(&d_temp_storage, temp_storage_bytes);
                cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, stream);
            }

            if (d_value_buf != d_values.Current()){
                cudaMemcpy(d_value_buf, d_value_alt_buf, num_items * sizeof(Tv),
                           cudaMemcpyDeviceToDevice);
            }
        } else {
            if (descending) {
                cub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes, d_keys, num_items, begin_bit, end_bit, stream);

                cudaMalloc(&d_temp_storage, temp_storage_bytes);
                cub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes, d_keys, num_items, begin_bit, end_bit, stream);
            } else {
                cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_items, begin_bit, end_bit, stream);

                cudaMalloc(&d_temp_storage, temp_storage_bytes);
                cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_items, begin_bit, end_bit, stream);
            }
        }

        if (d_key_buf != d_keys.Current()){
            cudaMemcpy(d_key_buf, d_key_alt_buf, num_items * sizeof(Tk),
                       cudaMemcpyDeviceToDevice);
        }
    }
};

extern "C" {

#define WRAP(Fn, Tk, Tv)                        \
void                                            \
radixsort_ ## Fn(   unsigned  num_items,        \
                    Tk  *d_key_buf,             \
                    Tk  *d_key_alt_buf,         \
                    Tv  *d_value_buf,           \
                    Tv  *d_value_alt_buf,       \
                    cudaStream_t stream,        \
                    int descending,             \
                    unsigned begin_bit,         \
                    unsigned end_bit      ) {   \
    RadixSort<Tk, Tv>::sort(num_items,          \
                            d_key_buf,          \
                            d_key_alt_buf,      \
                            d_value_buf,        \
                            d_value_alt_buf,    \
                            stream,             \
                            descending,         \
                            begin_bit,          \
                            end_bit);           \
}

WRAP(float, float, unsigned)
WRAP(double, double, unsigned)
WRAP(int32, int32_t, unsigned)
WRAP(uint32, uint32_t, unsigned)
WRAP(int64, int64_t, unsigned)
WRAP(uint64, uint64_t, unsigned)

#undef WRAP
} // end extern "C"
