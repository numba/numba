from __future__ import print_function, absolute_import
import os
from collections import namedtuple
from ctypes import c_int
from contextlib import contextmanager
import numpy as np
from numbapro import cuda

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
    return os.path.join(os.path.dirname(__file__), 'radixsort_details', fname)


def get_libradix_image(cc):
    with open(get_libradix_path(address_size, cc), 'rb') as fin:
        return fin.read()


BUCKET_SIZE = 256
SCAN_BLOCK_SUM_BLOCK_SIZE = 256


class Radixsort(object):
    def __init__(self, dtype, stream=0):
        self.context = cuda.current_context()
        self.cc = self.context.device.compute_capability
        self.dtype = dtype
        self.stream = stream

        itemsize = np.dtype(self.dtype).itemsize

        databitsize = itemsize * 8
        # initialize cuda functions
        image = get_libradix_image(cc=self.cc)
        self.module = self.context.create_module_image(image)
        self.cu_build_hist = self.module.get_function('cu_build_histogram')
        self.cu_scan_hist = self.module.get_function('cu_scan_histogram')
        self.cu_scan_bucket_index = self.module.get_function(
            'cu_scan_bucket_index')
        self.cu_compute_indices = self.module.get_function(
            'cu_compute_indices_uint{:d}'.format(databitsize))
        self.cu_scan_bucket_index = self.module.get_function(
            'cu_scan_bucket_index')
        self.cu_scatter = self.module.get_function('cu_scatter')

        self.cu_blockwise_sort = self.module.get_function(
            'cu_blockwise_sort_uint{:d}'.format(databitsize))

        self.flip_sign = False
        self.flip_float = False

        if dtype == np.dtype(np.int32) or dtype == np.dtype(np.int64):
            self.cu_sign_fix = self.module.get_function(
                'cu_sign_fix_uint{:d}'.format(databitsize))
            self.flip_sign = True

        elif dtype == np.dtype(np.float32):
            self.cu_float_to_uint = self.module.get_function('cu_float_to_uint')
            self.cu_uint_to_float = self.module.get_function('cu_uint_to_float')
            self.flip_float = True

        elif dtype == np.dtype(np.float64):
            self.cu_float_to_uint = self.module.get_function(
                'cu_double_to_uint')
            self.cu_uint_to_float = self.module.get_function(
                'cu_uint_to_double')
            self.flip_float = True

        else:
            if dtype not in [np.dtype(np.uint32), np.dtype(np.uint64)]:
                raise TypeError("{} is not supported".format(dtype))


    def _sentry_data(self, data):
        if data.dtype != self.dtype:
            raise TypeError("Mismatch dtype")
        if data.ndim != 1:
            raise ValueError("Can only sort one dimensional data")
        if data.strides[0] != data.dtype.itemsize:
            raise ValueError("Data must be C contiguous")

    def sort(self, data, reversed=False):
        self._sentry_data(data)

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

        with self._fix_bit_pattern(blkcount, d_data, count):
            # Sort loop
            for offset in range(stride):
                self._sortpass(d_data, d_sorted, d_hist, d_bucktotal, d_indices,
                               count, stride, offset, blkcount)
                # Swap data and sorted
                d_data, d_sorted = d_sorted, d_data

        # Prepare result
        if data_conv:
            d_data.copy_to_host(data, stream=self.stream)

    def select(self, data, seln):
        """
        Peform a partial sort to select the first N element after sorting.
        """
        self._sentry_data(data)

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

        with self._fix_bit_pattern(blkcount, d_data, count):
            # Start from MSB
            offset = stride - 1

            # Partition into small sub-block that we will finish off with a
            # subblock sort
            subblocks = []
            self._select(d_data, d_sorted, d_hist, d_bucktotal, d_indices,
                         count,
                         stride, offset, blkcount, seln, subblocks)

            # Sort sublocks
            if subblocks:
                subblocks = np.array(subblocks, order='F', dtype='uint32')
                arr_begin = np.ascontiguousarray(subblocks[:, 0])
                arr_count = np.ascontiguousarray(subblocks[:, 1])
                self._subblock_sort(d_data,
                                    cuda.to_device(arr_begin,
                                                   stream=self.stream),
                                    cuda.to_device(arr_count,
                                                   stream=self.stream))

        # Prepare result
        if data_conv:
            d_data.copy_to_host(data, stream=self.stream)

    def _subblock_sort(self, d_data, d_begin, d_count):
        confed = self.cu_blockwise_sort.configure((d_count.size,), (128,),
                                                  stream=self.stream)
        confed(d_data, d_begin, d_count)

    def _sortpass(self, d_data, d_sorted, d_hist, d_bucktotal, d_indices,
                  count, stride, offset, blkcount, h_bucktotal=None):

        kerns = self._configure(blkcount)

        kerns.build_hist(d_data, d_hist, c_int(stride), c_int(offset),
                         c_int(count))

        kerns.scan_hist(d_hist, d_bucktotal, c_int(blkcount))

        if h_bucktotal is not None:
            d_bucktotal.copy_to_host(h_bucktotal, stream=self.stream)

        kerns.scan_bucket(d_bucktotal)

        kerns.indexing(d_data, d_indices, d_hist, d_bucktotal, c_int(count),
                       c_int(offset))

        kerns.scatter(d_data, d_sorted, d_indices, c_int(count), c_int(stride))

    def _select(self, d_data, d_sorted, d_hist, d_bucktotal, d_indices,
                count, stride, offset, blkcount, seln, subblocks, substart=0,
                h_bucktotal=None):
        assert offset >= 0
        if h_bucktotal is None:
            h_bucktotal = cuda.pinned_array(d_bucktotal.shape,
                                            dtype=d_bucktotal.dtype)

        # Sort the MSB
        self._sortpass(d_data, d_sorted, d_hist, d_bucktotal, d_indices,
                       count, stride, offset, blkcount, h_bucktotal=h_bucktotal)

        # Get bucket total
        if self.stream:
            self.stream.synchronize()
        bucktotal = h_bucktotal.tolist()

        # Swap buffers
        # cuda.driver.device_to_device(d_data, d_sorted, d_sorted.alloc_size,
        #                              stream=self.stream)
        d_data.copy_to_device(d_sorted)

        # Find blocks that are too big and recursively partition it
        threshold = 128
        begin = 0
        for count in bucktotal:
            if count > threshold:
                # Recursively partition the block
                sub_blkcount = self._calc_block_count(count)
                sub_data = d_data[begin:begin + count]
                sub_sorted = d_sorted[begin:begin + count]
                self._select(sub_data,
                             sub_sorted,
                             d_hist, d_bucktotal, d_indices, count,
                             stride, offset - 1, sub_blkcount,
                             seln, subblocks, begin, h_bucktotal)
            elif count > 1:
                # Remember the small subblock
                # We might sort it later
                subblocks.append((begin + substart, count))

            begin += count
            if begin >= seln:
                break

    def _calc_block_count(self, count):
        return (count + BUCKET_SIZE - 1) // BUCKET_SIZE

    def _configure(self, blkcount):
        build_hist = self.cu_build_hist.configure((blkcount,), (BUCKET_SIZE,),
                                                  stream=self.stream)
        scan_hist = self.cu_scan_hist.configure((BUCKET_SIZE,),
                                                (SCAN_BLOCK_SUM_BLOCK_SIZE,),
                                                stream=self.stream)
        scan_bucket = self.cu_scan_bucket_index.configure((1,), (BUCKET_SIZE,),
                                                          stream=self.stream)
        indexing = self.cu_compute_indices.configure((blkcount,), (1,),
                                                     stream=self.stream)
        scatter = self.cu_scatter.configure((blkcount,), (BUCKET_SIZE,),
                                            stream=self.stream)

        return _select_kernels(build_hist, scan_hist, scan_bucket, indexing,
                               scatter)

    @contextmanager
    def _fix_bit_pattern(self, blkcount, d_data, count):
        if self.flip_float:
            cu_float_to_uint = self.cu_float_to_uint.configure((blkcount,),
                                                               (BUCKET_SIZE,),
                                                               stream=self.stream)
            cu_uint_to_float = self.cu_uint_to_float.configure((blkcount,),
                                                               (BUCKET_SIZE,),
                                                               stream=self.stream)

            cu_float_to_uint(d_data, d_data, c_int(count))

            yield

            cu_uint_to_float(d_data, d_data, c_int(count))

        elif self.flip_sign:

            cu_sign_fix = self.cu_sign_fix.configure((blkcount,),
                                                     (BUCKET_SIZE,),
                                                     stream=self.stream)

            cu_sign_fix(d_data, c_int(count))

            yield

            cu_sign_fix(d_data, c_int(count))

        else:
            yield


_select_kernels = namedtuple("select_kernels",
                             ['build_hist', 'scan_hist', 'scan_bucket',
                              'indexing', 'scatter'])
