from ctypes import *
import os
from contextlib import contextmanager
from numba.cuda.cudadrv.driver import device_pointer
from numba.cuda.cudadrv.drvapi import cu_stream
from numba.cuda.cudadrv.devicearray import auto_device, is_cuda_ndarray
from numba import cuda
import numpy as np

libname = 'radixsort.so'
libpath = os.path.join(os.path.dirname(__file__), 'details', libname)

lib = CDLL(libpath)

_argtypes = [
    c_void_p, # temp
    c_uint, # count
    c_void_p, # d_key
    c_void_p, # d_key_alt
    c_void_p, # d_vals
    c_void_p, # d_vals_alt
    cu_stream,
    c_int, # descending
    c_uint, # begin_bit
    c_uint, # end_bit
]

_overloads = {
    np.float32: 'float',
    np.float64: 'double',
    np.int32: 'int32',
    np.uint32: 'uint32',
    np.int64: 'int64',
    np.uint64: 'uint64'
}

overloads = {}


def _init():
    for ty, name in _overloads.items():
        dtype = np.dtype(ty)
        fn = getattr(lib, "radixsort_{}".format(name))
        overloads[dtype] = fn
        fn.argtypes = _argtypes
        fn.restype = c_void_p


_init()

lib.radixsort_cleanup.argtypes = [c_void_p]


def _devptr(p):
    if p is None:
        return None
    else:
        return device_pointer(p)


@contextmanager
def _autodevice(ary, stream, firstk=None):
    if ary is not None:
        dptr, conv = auto_device(ary, stream=stream)
        yield dptr
        if conv:
            if firstk is None:
                dptr.copy_to_host(ary, stream=stream)
            else:
                dptr.bind(stream)[:firstk].copy_to_host(ary[:firstk],
                                                        stream=stream)
    else:
        yield None


@cuda.autojit
def _cu_arange(ary, count):
    i = cuda.grid(1)
    if i < count:
        ary[i] = i


class RadixSort(object):
    def __init__(self, maxcount, dtype, descending=False, stream=0):
        self.maxcount = maxcount
        self.dtype = np.dtype(dtype)
        self._arysize = self.maxcount * self.dtype.itemsize
        self.descending = descending
        self.stream = stream
        self._sort = overloads[self.dtype]
        self._cleanup = lib.radixsort_cleanup

        ctx = cuda.current_context()
        self._temp_keys = ctx.memalloc(self._arysize)
        self._temp_vals = ctx.memalloc(self._arysize)
        self._temp = self._call(temp=None, keys=None, vals=None)

    def close(self):
        self._cleanup(self._temp)

    def _call(self, temp, keys, vals, begin_bit=0, end_bit=None):
        stream = self.stream.handle if self.stream else self.stream
        begin_bit = begin_bit
        end_bit = end_bit or self.dtype.itemsize * 8
        descending = int(self.descending)

        count = self.maxcount
        if keys:
            count = keys.size

        return self._sort(
            temp,
            c_uint(count),
            _devptr(keys),
            _devptr(self._temp_keys),
            _devptr(vals),
            _devptr(self._temp_vals),
            stream,
            descending,
            begin_bit,
            end_bit
        )

    def _sentry(self, ary):
        if ary.dtype != self.dtype:
            raise TypeError("dtype mismatch")
        if ary.size > self.maxcount:
            raise ValueError("keys array too long")

    def sort(self, keys, vals=None, begin_bit=0, end_bit=None):
        self._sentry(keys)
        with _autodevice(keys, self.stream) as d_keys:
            with _autodevice(vals, self.stream) as d_vals:
                self._call(self._temp, keys=d_keys, vals=d_vals,
                           begin_bit=begin_bit, end_bit=end_bit)

    def select(self, k, keys, vals=None, begin_bit=0, end_bit=None):
        self._sentry(keys)
        with _autodevice(keys, self.stream, firstk=k) as d_keys:
            with _autodevice(vals, self.stream, firstk=k) as d_vals:
                self._call(self._temp, keys=d_keys, vals=d_vals,
                           begin_bit=begin_bit, end_bit=end_bit)

    def init_arg(self, size):
        d_vals = cuda.device_array(size, dtype=np.uint32, stream=self.stream)
        _cu_arange.forall(d_vals.size, stream=self.stream)(d_vals, size)
        return d_vals

    def argselect(self, k, keys, begin_bit=0, end_bit=None):
        d_vals = self.init_arg(keys.size)
        self.select(k, keys, vals=d_vals, begin_bit=begin_bit, end_bit=end_bit)
        res = d_vals.bind(self.stream)[:k]
        if not is_cuda_ndarray(keys):
            res = res.copy_to_host(stream=self.stream)
        return res

    def argsort(self, keys, begin_bit=0, end_bit=None):
        d_vals = self.init_arg(keys.size)
        self.sort(keys, vals=d_vals, begin_bit=begin_bit, end_bit=end_bit)
        res = d_vals
        if not is_cuda_ndarray(keys):
            res = res.copy_to_host(stream=self.stream)
        return res


def test():
    keys = np.array(list(reversed(list(range(10 ** 3)))),
                    dtype=np.float32)
    orig = keys.copy()
    vals = np.arange(keys.size, dtype=np.uint32)

    # print(keys)
    # print(vals)

    rs = RadixSort(keys.size, dtype=keys.dtype, descending=True)

    k = 10
    vals = rs.argselect(k, keys)
    print(vals)

    assert np.all(orig.argsort()[::-1][:k] == vals)

    assert np.all(orig == keys)


if __name__ == '__main__':
    test()
