#include <cub/device/device_radix_sort.cuh>
#include <stdint.h>

struct TempStorage{
    void * storage;
    size_t storage_bytes;
};

static
void cleanup(TempStorage *ptr) {
    cudaFree(ptr->storage);
    delete ptr;
}

template <class Tk, class Tv=unsigned>
struct RadixSort {


    static
    TempStorage* sort(  TempStorage *temp,
                        unsigned  num_items,
                        Tk  *d_key_buf,
                        Tk  *d_key_alt_buf,
                        Tv  *d_value_buf,
                        Tv  *d_value_alt_buf,
                        cudaStream_t stream,
                        int descending,
                        unsigned begin_bit,
                        unsigned end_bit      )
    {
        cub::DoubleBuffer<Tk> d_keys(d_key_buf, d_key_alt_buf);
        if (temp == 0) {
            temp = new TempStorage;
            temp->storage = 0;
            temp->storage_bytes = 0;
        }
        if (d_value_buf) {
            // Sort KeyValue pairs
            cub::DoubleBuffer<Tv> d_values(d_value_buf, d_value_alt_buf);
            if (descending) {
                cub::DeviceRadixSort::SortPairsDescending(temp->storage,
                                                          temp->storage_bytes,
                                                          d_keys,
                                                          d_values,
                                                          num_items,
                                                          begin_bit,
                                                          end_bit,
                                                          stream);
            } else {
                cub::DeviceRadixSort::SortPairs(  temp->storage,
                                                  temp->storage_bytes,
                                                  d_keys,
                                                  d_values,
                                                  num_items,
                                                  begin_bit,
                                                  end_bit,
                                                  stream    );
            }

            if (temp->storage && d_value_buf != d_values.Current()){
                cudaMemcpyAsync(d_value_buf, d_value_alt_buf,
                                num_items * sizeof(Tv),
                                cudaMemcpyDeviceToDevice,
                                stream);
            }
        } else {
            // Sort Keys only
            if (descending) {
                cub::DeviceRadixSort::SortKeysDescending(   temp->storage,
                                                            temp->storage_bytes,
                                                            d_keys,
                                                            num_items,
                                                            begin_bit,
                                                            end_bit,
                                                            stream  );
            } else {
                cub::DeviceRadixSort::SortKeys( temp->storage,
                                                temp->storage_bytes,
                                                d_keys,
                                                num_items,
                                                begin_bit,
                                                end_bit,
                                                stream  );
            }
        }

        if (temp->storage && d_key_buf != d_keys.Current()){
            cudaMemcpyAsync(d_key_buf, d_key_alt_buf, num_items * sizeof(Tk),
                            cudaMemcpyDeviceToDevice, stream);
        }

        if (temp->storage == 0) {
            cudaMalloc(&temp->storage, temp->storage_bytes);
            return temp;
        }
        return temp;
    }
};

extern "C" {

#define WRAP(Fn, Tk, Tv)                        \
void                                            \
radixsort_ ## Fn(   TempStorage *temp,          \
                    unsigned  num_items,        \
                    Tk  *d_key_buf,             \
                    Tk  *d_key_alt_buf,         \
                    Tv  *d_value_buf,           \
                    Tv  *d_value_alt_buf,       \
                    cudaStream_t stream,        \
                    int descending,             \
                    unsigned begin_bit,         \
                    unsigned end_bit      ) {   \
    RadixSort<Tk, Tv>::sort(temp,               \
                            num_items,          \
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

void
radixsort_cleanup(TempStorage *ptr) {
    cleanup(ptr);
}

#undef WRAP
} // end extern "C"
