from ctypes import *
import os
from contextlib import contextmanager
from numba.cuda.cudadrv.driver import device_pointer
from numba.cuda.cudadrv.drvapi import cu_stream
from numba.cuda.cudadrv.devicearray import auto_device, is_cuda_ndarray
from numba import cuda
import numpy as np

libname = 'mgpusort.so'
libpath = os.path.join(os.path.dirname(__file__), 'details', libname)

lib = CDLL(libpath)

_argtypes = [
    # d_key
    c_void_p,
    # d_vals
    c_void_p,
    # N
    c_uint,
    # segments
    c_void_p,
    # Nseg
    c_uint,
    # stream
    cu_stream,
]

_overloads = [
    'float32',
    'float64',
]

overloads = {}


def init():
    for t in _overloads:
        fn = getattr(lib, 'segsortpairs_{}'.format(t))
        fn.argtypes = _argtypes
        overloads[np.dtype(t)] = fn


init()


@contextmanager
def _autodevice(ary, stream):
    if ary is not None:
        dptr, conv = auto_device(ary, stream=stream)
        yield dptr
        if conv:
            dptr.copy_to_host(ary, stream=stream)
    else:
        yield None


def _segmentedsort(d_keys, d_vals, d_segments, stream):
    overloads[d_keys.dtype](device_pointer(d_keys),
                            device_pointer(d_vals),
                            d_keys.size,
                            device_pointer(d_segments),
                            d_segments.size,
                            stream.handle if stream else 0)


def segmented_sort(keys, vals, segments, stream=0):
    with _autodevice(keys, stream) as d_keys:
        with _autodevice(vals, stream) as d_vals:
            d_segments, _ = auto_device(segments, stream=stream)
            _segmentedsort(d_keys, d_vals, d_segments, stream)


def test():
    keys = np.array(list(reversed(range(1000))), dtype=np.float32)
    vals = np.arange(keys.size, dtype=np.uint32)
    segments = np.array([500], dtype=np.int32)
    print(keys)
    s = 0

    d_keys = cuda.to_device(keys)
    d_vals = cuda.to_device(vals)
    segmented_sort(d_keys, d_vals, segments, stream=s)
    # s.synchronize()
    print(d_keys.copy_to_host())
    print(d_vals.copy_to_host())


if __name__ == '__main__':
    test()
