"""
Uses segmented sort implementation from ModernGPU which has the following
license:

Copyright (c) 2013, NVIDIA CORPORATION.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the NVIDIA CORPORATION nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import print_function, absolute_import, division
import ctypes
import os
from contextlib import contextmanager
from numba.cuda.cudadrv.driver import device_pointer
from numba.cuda.cudadrv.drvapi import cu_stream
from numba.cuda.cudadrv.devicearray import auto_device
from numba import findlib
import numpy as np
from numbapro.findlib import library_extension

libname = 'nbpro_segsort.' + library_extension()
libpath = os.path.join(findlib.get_lib_dir(), libname)

lib = ctypes.CDLL(libpath)

_argtypes = [
    # d_key
    ctypes.c_void_p,
    # d_vals
    ctypes.c_void_p,
    # N
    ctypes.c_uint,
    # segments
    ctypes.c_void_p,
    # Nseg
    ctypes.c_uint,
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

