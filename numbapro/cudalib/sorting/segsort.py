from ctypes import *
import os
from contextlib import contextmanager
from numba.cuda.cudadrv.driver import device_pointer
from numba.cuda.cudadrv.drvapi import cu_stream
from numba.cuda.cudadrv.devicearray import auto_device
from numba import findlib
import numpy as np

libname = 'nbpro_segsort.so'
libpath = os.path.join(findlib.get_lib_dir(), libname)

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

_support_types = {
    np.float32: 'float32',
    np.float64: 'float64',
    np.int32: 'int32',
    np.uint32: 'uint32',
    np.int64: 'int64',
    np.uint64: 'uint64'
}

_overloads = {}


def _init():
    for k, v in _support_types.items():
        fn = getattr(lib, 'segsortpairs_{0}'.format(v))
        fn.argtypes = _argtypes
        _overloads[np.dtype(k)] = fn


_init()


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
    _overloads[d_keys.dtype](device_pointer(d_keys),
                             device_pointer(d_vals),
                             d_keys.size,
                             device_pointer(d_segments),
                             d_segments.size,
                             stream.handle if stream else 0)


def segmented_sort(keys, vals, segments, stream=0):
    """Perform a inplace sort on small segments (N < 1e6).

    Args
    ----
    keys : 1d array
        Keys to sort inplace.

    vals: 1d array
        Values to be reordered inplace along the sort. Only support uint32.

    segments: 1d array of int32
        Segments separation location. e.g. array([3, 6, 8]) for segment of
        keys[:3], keys[3:6], keys[6:8], keys[8:].

    stream: cuda stream; optional
        A cuda stream in which the kernels are executed.
    """
    with _autodevice(keys, stream) as d_keys:
        with _autodevice(vals, stream) as d_vals:
            d_segments, _ = auto_device(segments, stream=stream)
            _segmentedsort(d_keys, d_vals, d_segments, stream)

