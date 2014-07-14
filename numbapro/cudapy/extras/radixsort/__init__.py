from __future__ import print_function, absolute_import
import os
import numpy as np
from numbapro import cuda
from ctypes import c_int

# CPU address size
address_size = tuple.__itemsize__ * 8

# List supported compute capability from high to low
supported_cc = (3, 5), (3, 0), (2, 0)


def _normalize_cc(cc):
    for ss in supported_cc:
        if cc >= ss:
            return ss


def get_libradix_path(bits, cc):
    cc = _normalize_cc(cc)
    fname = 'libradix%d_sm%u%u.ptx' % (bits, cc[0], cc[1])
    return os.path.join(os.path.dirname(__file__), fname)


def get_libradix_image(cc):
    with open(get_libradix_path(address_size, cc), 'rb') as fin:
        return fin.read()


BUCKET_SIZE = 256
SCAN_BLOCK_SUM_BLOCK_SIZE = 256


class Radixsort(object):
    def __init__(self, dtype):
        self.context = cuda.current_context()
        self.cc = self.context.device.compute_capability
        self.dtype = dtype
        itemsize = np.dtype(self.dtype).itemsize

        # initialize cuda functions
        image = get_libradix_image(cc=self.cc)
        self.module = self.context.create_module_image(image)
        self.cu_build_hist = self.module.get_function('cu_build_histogram')
        self.cu_scan_hist = self.module.get_function('cu_scan_histogram')
        self.cu_scan_bucket_index = self.module.get_function(
            'cu_scan_bucket_index')
        self.cu_compute_indices = self.module.get_function(
            'cu_compute_indices_uint%u' % (itemsize * 8))
        self.cu_scan_bucket_index = self.module.get_function(
            'cu_scan_bucket_index')
        self.cu_scatter = self.module.get_function('cu_scatter')

    def sort(self, data):
        if data.dtype != self.dtype:
            raise TypeError("Mismatch dtype")
        if data.ndim != 1:
            raise ValueError("Can only sort one dimensional data")
        if data.strides[0] != data.dtype.itemsize:
            raise ValueError("Data must be C contiguous")

        dtype = data.dtype
        idx_dtype = np.dtype(np.uint32)
        stride = data.strides[0]
        count = data.size
        blkcount = self._calc_block_count(count)
        # Initialize device memory
        d_indices = cuda.device_array(count, dtype=idx_dtype)
        d_sorted = cuda.device_array(count, dtype=dtype)
        d_hist = cuda.device_array(blkcount * BUCKET_SIZE, dtype=idx_dtype)
        d_bucktotal = cuda.device_array(BUCKET_SIZE, dtype=idx_dtype)

        d_data, data_conv = cuda.devicearray.auto_device(data)

        # Configure all kernels
        (build_hist, scan_hist,
         scan_bucket, indexing, scatter) = self._configure(blkcount)

        # Sort loop
        for offset in range(stride):
            build_hist(d_data, d_hist, c_int(stride), c_int(offset),
                       c_int(count))

            scan_hist(d_hist, d_bucktotal, c_int(blkcount))

            scan_bucket(d_bucktotal)

            indexing(d_data, d_indices, d_hist, d_bucktotal, c_int(count),
                     c_int(offset))

            scatter(d_data, d_sorted, d_indices, c_int(count), c_int(stride))

            # Swap data and sorted
            d_data, d_sorted = d_sorted, d_data


        # Prepare result
        if data_conv:
            d_data.copy_to_host(data)

    def _calc_block_count(self, count):
        return (count + BUCKET_SIZE - 1) // BUCKET_SIZE

    def _configure(self, blkcount):
        build_hist = self.cu_build_hist.configure((blkcount,), (BUCKET_SIZE,))
        scan_hist = self.cu_scan_hist.configure((BUCKET_SIZE,),
                                                (SCAN_BLOCK_SUM_BLOCK_SIZE,))
        scan_bucket = self.cu_scan_bucket_index.configure((1,), (BUCKET_SIZE,))
        indexing = self.cu_compute_indices.configure((blkcount,), (1,))
        scatter = self.cu_scatter.configure((blkcount,), (BUCKET_SIZE,))
        return build_hist, scan_hist, scan_bucket, indexing, scatter
