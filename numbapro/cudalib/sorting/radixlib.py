from ctypes import *
import os
from numba.cuda.cudadrv.driver import device_pointer
from numba.cuda.cudadrv.drvapi import cu_stream
from numba.cuda.cudadrv.devicearray import auto_device
from numba import cuda
import numpy as np

libname = 'radixsort.so'
libpath = os.path.join(os.path.dirname(__file__), 'radixsort_details', libname)

lib = CDLL(libpath)

_argtypes = [
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
for ty, name in _overloads.items():
    dtype = np.dtype(ty)
    fn = getattr(lib, "radixsort_{}".format(name))
    overloads[dtype] = fn
    fn.argtypes = _argtypes


def _radixsortpairs(keys, vals=None, descending=False, stream=0,
                    begin_bit=0, end_bit=None):
    assert not vals or vals.dtype == np.dtype(np.uint32)
    ctx = cuda.current_context()

    keys_alt = ctx.memalloc(keys.alloc_size)
    if vals is not None:
        vals_alt = ctx.memalloc(vals.alloc_size)

    count = keys.size

    stream = stream.handle if stream else stream
    begin_bit = begin_bit
    end_bit = end_bit or keys.dtype.itemsize * 8
    descending = int(descending)

    overloads[keys.dtype](
        c_uint(count),
        device_pointer(keys),
        device_pointer(keys_alt),
        device_pointer(vals) if vals is not None else None,
        device_pointer(vals_alt) if vals is not None else None,
        stream,
        descending,
        begin_bit,
        end_bit
    )


def radixsortpairs(keys, vals=None, descending=False, stream=0):
    d_keys, conv_keys = auto_device(keys, stream=stream)
    if vals:
        d_vals, conv_vals = auto_device(vals, stream=stream)
        _radixsortpairs(d_keys, vals=d_vals, descending=descending,
                        stream=stream)
        if conv_vals:
            vals.copy_to_host(vals, stream=stream)
    else:
        _radixsortpairs(d_keys, descending=descending, stream=stream)

    if conv_keys:
        keys.copy_to_host(keys, stream=stream)


def radix_select(keys, k, descending=False, stream=0, retindex=False,
                 storeidx=None):
    d_keys, conv_keys = auto_device(keys, stream=stream)
    if retindex:
        vals = np.arange(d_keys.size, dtype=np.uint32)
        d_vals = cuda.to_device(vals, stream=stream)
    else:
        d_vals = None

    radixsortpairs(d_keys, d_vals, descending=descending, stream=stream)

    if conv_keys:
        d_keys[:k].copy_to_host(keys[:k], stream=stream)
        if retindex:
            return d_vals[:k].copy_to_host(stream=stream)
    else:
        if retindex:
            if storeidx:
                storeidx.copy_to_device(d_vals[:k], stream=stream)
            return d_vals[:k]


def radix_argselect(keys, k, descending=False, stream=0, storeidx=None):
    return radix_select(keys, k, descending=descending, stream=stream,
                        retindex=True, storeidx=storeidx)

#
# def test():
#     import numpy as np
#
#     keys = np.array(list(reversed(list(range(32 * 10 ** 4)))), dtype=np.float32)
#     vals = np.arange(keys.size, dtype=np.uint32)
#
#     print(keys)
#     print(vals)
#
#     d_keys = cuda.to_device(keys)
#     # d_vals = cuda.to_device(vals)
#
#     radixsortpairs(d_keys) #, d_vals)
#
#     print(d_keys.copy_to_host())
#     # print(d_vals.copy_to_host())


def test():
    keys = np.array(list(reversed(list(range(32 * 10 ** 4)))), dtype=np.float32)
    vals = np.arange(keys.size, dtype=np.uint32)

    print(keys)
    print(vals)

    d_keys = cuda.to_device(keys)

    k = 10
    d_idx = radix_argselect(d_keys, k=k)

    print(d_keys[:k].copy_to_host())
    idx = d_idx.copy_to_host()
    print(idx)
    print(keys[idx])


if __name__ == '__main__':
    test()
