from __future__ import print_function, absolute_import, division
import os
import copy
import math
from collections import namedtuple
from ctypes import c_uint
from contextlib import contextmanager
from timeit import default_timer as timer
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


class _TempStorage(object):
    def __init__(self, dtype, count, stream):
        dtype = np.dtype(dtype)
        idx_dtype = np.dtype(np.uint32)
        self.stride = dtype.itemsize
        self.count = count
        self.blkcount = _calc_block_count(self.count)
        # Initialize device memory
        self.d_indices = cuda.device_array(self.count, dtype=idx_dtype,
                                           stream=stream)
        self.d_sorted = cuda.device_array(self.count, dtype=dtype,
                                          stream=stream)
        self.d_hist = cuda.device_array(self.blkcount * BUCKET_SIZE,
                                        dtype=idx_dtype,
                                        stream=stream)
        self.d_bucktotal = cuda.device_array(BUCKET_SIZE, dtype=idx_dtype,
                                             stream=stream)

    @contextmanager
    def prepare_data(self, data, stream):
        (self.d_data,
         self.data_conv) = cuda.devicearray.auto_device(data, stream=stream)
        yield
        if self.data_conv:
            self.d_data.copy_to_host(data, stream=stream)
        del self.d_data
        del self.data_conv


class _IndexStorage(object):
    def __init__(self, count, dtype, stream):
        self.d_order = cuda.device_array(count, dtype=dtype, stream=stream)
        self.d_sorted = cuda.device_array(count, dtype=dtype, stream=stream)
        self.stride = self.d_order.dtype.itemsize

    def swap(self, stream):
        self.d_sorted, self.d_order = self.d_order, self.d_sorted

    def getresult(self):
        return self.d_order


class _SelectIndexStorage(_IndexStorage):
    def swap(self, stream):
        self.d_order.copy_to_device(self.d_sorted, stream=stream)

    def slice(self, start, stop, stream):
        d_sorted = self.d_sorted.getitem(slice(start, stop), stream=stream)
        d_order = self.d_order.getitem(slice(start, stop), stream=stream)
        thecopy = copy.copy(self)
        thecopy.d_sorted = d_sorted
        thecopy.d_order = d_order
        return thecopy


_select_states = namedtuple("select_states", ["ts", "sis", "hbt"])
_sort_states = namedtuple("sort_states", ["ts", "sis"])


def _calc_block_count(count, blksize=BUCKET_SIZE):
    return (count + blksize - 1) // blksize


SELECT_THRESHOLD = 10000


class Radixsort(object):
    """Radixsort based GPU sorting ans k-selection algorithm.

    Types
    -----
    Array of int32, uint32, int64, uint64, float32, float64 are supported.

    Functions
    ---------
    - sort, argsort
    - select, argselect

    The arg* version turns an index array similar to ``numpy.argsort``.

    Note
    -----
    Current implementation is limited to 32-bit due to actual hardware memory
    capacity.  CUDA GPU is limited to 12GB of RAM.  To sort a 4GB float32
    array, the algorithm allocates two 4GB of data array for scatter, one 4GB
    index array and other smaller array for histograms.  The 12GB RAM is not
    enough for data size of 4 billion payload.  Therefore, 32-bit uint32 is
    used as indices.  Until a higher capability GPU is available, a 64-bit
    implementation will not be usable because 64-bit indices will occupy twice
    as much space.


    """

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

        self.cu_blockwise_argsort = self.module.get_function(
            'cu_blockwise_argsort_uint{:d}'.format(databitsize))

        self.cu_singleblock_argsort = self.module.get_function(
            'cu_singleblock_argsort_uint{:d}'.format(databitsize))

        self.cu_singleblock_sort = self.module.get_function(
            'cu_singleblock_sort_uint{:d}'.format(databitsize))

        self.cu_invert = self.module.get_function(
            'cu_invert_uint{:d}'.format(databitsize))

        self.cu_index_init = self.module.get_function('cu_arange_uint32')

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

    def batch_sort(self, dtype, count, reverse=False, getindices=False):
        states = self._init_sort_states(dtype, count, self.stream, getindices)

        def _next(data):
            return self.sort(data, reverse=reverse, getindices=getindices,
                             _states=states)

        return _next

    def batch_argsort(self, dtype, count, reverse=False):
        return self.batch_sort(dtype, count, reverse=reverse, getindices=True)

    def sort(self, data, reverse=False, getindices=False, _states=None):
        """Perform inplace sort on the data.

        Args
        ----
        - data: cpu/gpu array
        - reverse: bool
            Optional.  Set to True to reverse the order of the sort
        - getindices: bool
            Optional.  Set to True to return the sorted indices
        """
        self._sentry_data(data)
        states = _states or self._init_sort_states(data.dtype, data.size,
                                                   self.stream, getindices)
        del _states   # just to avoid typos

        ts = states.ts
        with ts.prepare_data(data, self.stream):
            indstore = None

            if getindices:
                indstore = states.sis
                conf = self.cu_index_init.configure((ts.blkcount,),
                                                    (BUCKET_SIZE,),
                                                    stream=self.stream)
                conf(indstore.d_order, c_uint(ts.count))

            with self._filter(reverse, ts):
                ts.d_data = self._sortloop(ts.d_data, ts.d_sorted, ts.d_hist,
                                           ts.d_bucktotal, ts.d_indices,
                                           ts.count, ts.stride, ts.blkcount,
                                           indstore)

            # Prepare result
            if indstore:
                output = indstore.getresult()
                return (output.copy_to_host(stream=self.stream)
                        if ts.data_conv else output)

    def argsort(self, data, reverse=False):
        """The same as ``sort(..., getindices=True)``
        """
        return self.sort(data, reverse=reverse, getindices=True)

    def _init_sort_states(self, dtype, count, stream, getindices):
        ts = _TempStorage(dtype, count, stream=self.stream)
        if getindices:
            sis = _SelectIndexStorage(ts.count, dtype=np.uint32,
                                      stream=self.stream)
        else:
            sis = None
        return _sort_states(ts=ts, sis=sis)

    def _init_select_states(self, dtype, count, stream, getindices):
        ts = _TempStorage(dtype, count, stream=self.stream)
        if getindices:
            sis = _SelectIndexStorage(ts.count, dtype=np.uint32,
                                      stream=self.stream)
        else:
            sis = None
        hbt = cuda.pinned_array(ts.d_bucktotal.shape,
                                dtype=ts.d_bucktotal.dtype)
        return _select_states(ts=ts, sis=sis, hbt=hbt)

    def batch_select(self, dtype, count, k, reverse=False, getindices=False):
        states = self._init_select_states(dtype, count, self.stream,
                                          getindices)

        def _next(data):
            return self.select(data, k=k, reverse=reverse,
                               getindices=getindices, _states=states)

        return _next

    def batch_argselect(self, dtype, count, k, reverse=False):
        return self.batch_select(dtype, count, k, reverse=reverse,
                                 getindices=True)

    def select(self, data, k, reverse=False, getindices=False, _states=None):
        """
        Perform a inplace partial sort to select the first k-element in
        sorted order.  Only the first k-elements are guaranteed to be
        valid.

        Args
        ----
        - data: cpu/gpu array
        - k: int
            Number of elements to select
        - reverse: bool
            Optional.  Reverse the order of the sort
        - getindices: bool
            Optional.  Set to True to return the sorted indices
        """
        self._sentry_data(data)
        states = _states or self._init_select_states(data.dtype, data.size,
                                                     self.stream, getindices)
        del _states  # just to prevent typos

        ts = states.ts
        h_bucktotal = states.hbt

        with ts.prepare_data(data, self.stream):
            indstore = None
            if getindices:
                indstore = states.sis
                conf = self.cu_index_init.configure((ts.blkcount,),
                                                    (BUCKET_SIZE,),
                                                    stream=self.stream)
                conf(indstore.d_order, c_uint(ts.count))

            with self._filter(reverse, ts):
                if ts.count < SELECT_THRESHOLD:
                    # Data set too small to use k-selection
                    # Fallback to sorting
                    ts.d_data = self._sortloop(ts.d_data, ts.d_sorted,
                                               ts.d_hist, ts.d_bucktotal,
                                               ts.d_indices, ts.count,
                                               ts.stride, ts.blkcount, indstore)

                else:
                    # Start from MSB
                    offset = ts.stride - 1

                    # Partition into small sub-block that we will finish off
                    # with a subblock sort
                    subblocks = []
                    self._select(ts.d_data, ts.d_sorted, ts.d_hist,
                                 ts.d_bucktotal,
                                 ts.d_indices, ts.count, ts.stride, offset,
                                 ts.blkcount, k, subblocks, indstore=indstore,
                                 h_bucktotal=h_bucktotal)

                    # Sort sub-blocks
                    if subblocks:
                        subblocks = np.array(subblocks, order='F',
                                             dtype=np.uint32)
                        arr_begin = np.ascontiguousarray(subblocks[:, 0])
                        arr_count = np.ascontiguousarray(subblocks[:, 1])

                        nsb_length = len(subblocks)
                        if nsb_length <= min(ts.d_hist.size,
                                             ts.d_bucktotal.size):
                            # Avoid extra allocation if we can
                            d_begin = ts.d_hist.getitem(slice(0, nsb_length),
                                                        stream=self.stream)
                            d_count = ts.d_bucktotal.getitem(
                                slice(0, nsb_length),
                                stream=self.stream)
                            d_begin.copy_to_device(arr_begin,
                                                   stream=self.stream)
                            d_count.copy_to_device(arr_count,
                                                   stream=self.stream)
                        else:
                            # Allocate new array for the sublock counts
                            d_begin = cuda.to_device(arr_begin,
                                                     stream=self.stream)
                            d_count = cuda.to_device(arr_count,
                                                     stream=self.stream)

                        self._subblock_sort(ts.d_data, d_begin, d_count,
                                            indstore=indstore)

            # Prepare result
            if indstore:
                output = indstore.getresult().getitem(slice(0, k),
                                                      stream=self.stream)
                return (output.copy_to_host(stream=self.stream)
                        if ts.data_conv else output)

    def argselect(self, data, k, reverse=False):
        """The same as ``select(..., getindices=True)``
        """
        return self.select(data, k=k, reverse=reverse, getindices=True)

    def _sortloop(self, d_data, d_sorted, d_hist, d_bucktotal, d_indices,
                  count, stride, blkcount, indstore=None):

        SORT_THRESHOLD = 256 * 4  # Must match code
        if count < SORT_THRESHOLD:

            if indstore is not None:
                conf = self.cu_singleblock_argsort.configure((1,), (256,),
                                                             stream=self.stream)
                conf(d_data, indstore.d_order, c_uint(count))
                return d_data

            else:
                conf = self.cu_singleblock_sort.configure((1,), (256,),
                                                          stream=self.stream)
                conf(d_data, c_uint(count))
                return d_data

        # Sort loop
        for offset in range(stride):
            self._sortpass(d_data, d_sorted, d_hist, d_bucktotal,
                           d_indices, count, stride, offset,
                           blkcount, indstore=indstore)
            # Swap data and sorted
            d_data, d_sorted = d_sorted, d_data

        return d_data

    def _subblock_sort(self, d_data, d_begin, d_count, indstore=None):
        TPB = 128   # Must match the kernel

        if indstore is not None:
            confed = self.cu_blockwise_argsort.configure((d_count.size,),
                                                         (TPB,),
                                                         stream=self.stream)
            confed(d_data, indstore.d_order, d_begin, d_count)
        else:
            confed = self.cu_blockwise_sort.configure((d_count.size,),
                                                      (TPB,),
                                                      stream=self.stream)
            confed(d_data, d_begin, d_count)

    def _sortpass(self, d_data, d_sorted, d_hist, d_bucktotal, d_indices,
                  count, stride, offset, blkcount, h_bucktotal=None,
                  evt_bucktotal=None, indstore=None):

        kerns = self._configure(blkcount)

        kerns.build_hist(d_data, d_hist, c_uint(stride), c_uint(offset),
                         c_uint(count))

        kerns.scan_hist(d_hist, d_bucktotal, c_uint(blkcount))

        if h_bucktotal is not None:
            d_bucktotal.copy_to_host(h_bucktotal, stream=self.stream)
            evt_bucktotal.record(self.stream)

        kerns.scan_bucket(d_bucktotal)

        kerns.indexing(d_data, d_indices, d_hist, d_bucktotal, c_uint(count),
                       c_uint(offset))

        kerns.scatter(d_data, d_sorted, d_indices, c_uint(count),
                      c_uint(stride))
        if indstore:
            # Sort the index table
            kerns.scatter(indstore.d_order, indstore.d_sorted,
                          d_indices, c_uint(count), c_uint(indstore.stride))
            indstore.swap(stream=self.stream)

    def _select(self, d_data, d_sorted, d_hist, d_bucktotal, d_indices,
                count, stride, offset, blkcount, seln, subblocks, substart=0,
                h_bucktotal=None, indstore=None):
        if offset == 0:
            # We reach here when the data have too many duplication.
            self._sortpass(d_data, d_sorted, d_hist, d_bucktotal,
                           d_indices, count, stride, offset,
                           blkcount, indstore=indstore)
            d_data.copy_to_device(d_sorted, stream=self.stream)
            return

        assert h_bucktotal is not None

        # Sort the MSB
        evt = cuda.event()
        self._sortpass(d_data, d_sorted, d_hist, d_bucktotal, d_indices,
                       count, stride, offset, blkcount,
                       h_bucktotal=h_bucktotal, evt_bucktotal=evt,
                       indstore=indstore)

        # Swap buffers
        d_data.copy_to_device(d_sorted, stream=self.stream)

        # Get bucket total
        evt.synchronize()
        bucktotal = h_bucktotal.tolist()

        # Find blocks that are too big and recursively partition it
        threshold = 128  # depends on CC and resource usage of
        # cu_blockwise_sort
        begin = 0
        for count in bucktotal:
            if count > threshold:
                # Recursively partition the block
                sub_blkcount = _calc_block_count(count)
                sub_data = d_data.getitem(slice(begin, begin + count),
                                          stream=self.stream)
                sub_sorted = d_sorted.getitem(slice(begin, begin + count),
                                              stream=self.stream)
                if indstore:
                    subindstore = indstore.slice(begin, begin + count,
                                                 stream=self.stream)
                else:
                    subindstore = None

                self._select(sub_data,
                             sub_sorted,
                             d_hist, d_bucktotal, d_indices, count,
                             stride, offset - 1, sub_blkcount,
                             seln, subblocks, begin, h_bucktotal,
                             indstore=subindstore)
            elif count > 1:
                # Remember the small subblock
                # We might sort it later
                subblocks.append((begin + substart, count))

            begin += count
            if begin >= seln:
                break

    def _configure(self, blkcount):
        build_hist = self.cu_build_hist.configure((blkcount,), (BUCKET_SIZE,),
                                                  stream=self.stream)
        scan_hist = self.cu_scan_hist.configure((BUCKET_SIZE,),
                                                (SCAN_BLOCK_SUM_BLOCK_SIZE,),
                                                stream=self.stream)
        scan_bucket = self.cu_scan_bucket_index.configure((1,), (BUCKET_SIZE,),
                                                          stream=self.stream)
        indexing = self.cu_compute_indices.configure((blkcount,), (64,),
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

            cu_float_to_uint(d_data, d_data, c_uint(count))

            yield

            cu_uint_to_float(d_data, d_data, c_uint(count))

        elif self.flip_sign:

            cu_sign_fix = self.cu_sign_fix.configure((blkcount,),
                                                     (BUCKET_SIZE,),
                                                     stream=self.stream)

            cu_sign_fix(d_data, c_uint(count))

            yield

            cu_sign_fix(d_data, c_uint(count))

        else:
            yield

    @contextmanager
    def _filter(self, reverse, ts):
        with self.context.trashing.defer_cleanup():
            with self._fix_bit_pattern(ts.blkcount, ts.d_data, ts.count):
                with self._inverted(reverse, ts.blkcount, ts.d_data, ts.count):
                    yield

    @contextmanager
    def _inverted(self, enabled, blkcount, d_data, count):
        if enabled:
            invert = self.cu_invert.configure((blkcount,), (BUCKET_SIZE,),
                                              stream=self.stream)
            invert(d_data, c_uint(count))
            yield
            invert(d_data, c_uint(count))
        else:
            yield


_select_kernels = namedtuple("select_kernels",
                             ['build_hist', 'scan_hist', 'scan_bucket',
                              'indexing', 'scatter'])


def benchmark_sort(dtype=np.float64, count=10 ** 6, getindices=False,
                   reverse=False, seed=None):
    """Radixsort library benchmark code.
    """
    if seed is not None:
        np.random.seed(seed)

    data = np.random.rand(count).astype(dtype)
    orig = data.copy()
    gold = data.copy()

    ts = timer()
    gold.sort()
    te = timer()
    cpu_time = te - ts

    if reverse:
        gold = gold[::-1]

    rs = Radixsort(data.dtype)

    # Do sort
    ts = timer()
    if getindices:
        indices = rs.argsort(data, reverse=reverse)
    else:
        indices = rs.sort(data, reverse=reverse)
    te = timer()
    gpu_time = te - ts

    # Check result
    assert np.all(data == gold)

    if getindices:
        assert (np.all(orig[indices] == gold))
    else:
        assert indices is None

    return cpu_time, gpu_time


def benchmark_select(dtype=np.float64, k=10, count=10 ** 6, getindices=False,
                     reverse=False, seed=None):
    if seed is not None:
        np.random.seed(seed)

    data = np.random.rand(count).astype(dtype)
    orig = data.copy()
    gold = data.copy()

    ts = timer()
    gold.sort()
    te = timer()
    cpu_time = te - ts

    if reverse:
        gold = gold[::-1]
    gold = gold[:k]
    rs = Radixsort(data.dtype)

    # Do sort
    ts = timer()
    if getindices:
        indices = rs.argselect(data, k=k, reverse=reverse)
    else:
        indices = rs.select(data, k=k, reverse=reverse)
    te = timer()
    gpu_time = te - ts

    data = data[:k]
    # check result
    assert np.all(data == gold)
    if getindices:
        assert np.all(orig[indices] == gold)
    else:
        assert indices is None

    return cpu_time, gpu_time
