import sys
from ctypes import *
import numpy as np
from numpy.ctypeslib import c_intp
from numbapro._cuda import driver as _cuda
from numbapro._utils.ndarray import *

_pyobject_head_fields = [('pyhead1', c_size_t),
                         ('pyhead2', c_void_p),]

if hasattr(sys, 'getobjects'):
    _pyobject_head_fields = [('pyhead3', c_int),
                             ('pyhead4', c_int),] + \
                              _pyobject_head_fields

_numpy_fields = _pyobject_head_fields + \
      [('data', c_void_p),                      # data
       ('nd',   c_int),                         # nd
       ('dimensions', POINTER(c_intp)),       # dimensions
       ('strides', POINTER(c_intp)),          # strides
        #  NOTE: The following fields are unused.
        #        Not sending to GPU to save transfer bandwidth.
        #       ('base', c_void_p),                      # base
        #       ('desc', c_void_p),                      # descr
        #       ('flags', c_int),                        # flags
        #       ('weakreflist', c_void_p),               # weakreflist
        #       ('maskna_dtype', c_void_p),              # maskna_dtype
        #       ('maskna_data', c_void_p),               # maskna_data
        #       ('masna_strides', POINTER(c_intp)),    # masna_strides
      ]

def ndarray_to_device_memory(ary, stream=0, copy=True, pinned=False):
    packed = ndarray_device_memory_and_data(ary, stream=stream, copy=copy,
                                            pinned=pinned)
    retr, struct, data_is_ignored = packed
    return retr, struct

def ndarray_device_memory_and_data(ary, stream=0, copy=True, pinned=False):
    retriever, gpu_data = ndarray_data_to_device_memory(ary,
                                                        stream=stream,
                                                        copy=copy,
                                                        pinned=pinned)
    gpu_struct = ndarray_device_memory_from_data(gpu_data,
                                                 ary.ctypes.shape,
                                                 ary.ctypes.strides,
                                                 stream=stream,
                                                 pinned=pinned)
    return retriever, gpu_struct, gpu_data

def ndarray_device_memory_from_data(gpu_data, c_shape, c_strides, stream=0,
                                    pinned=False):
    nd = len(c_shape)
    
    more_fields = [
        ('shape_array',   c_intp * nd),
        ('strides_array', c_intp * nd),
    ]

    class NumpyStructure(Structure):
        _fields_ = _numpy_fields + more_fields

    gpu_struct = _cuda.DeviceMemory(sizeof(NumpyStructure))

    # get base address of the GPU ndarray structure
    base_addr = gpu_struct._handle.value

    to_intp_p = lambda x: cast(c_void_p(x), POINTER(c_intp))

    # Offset to shape and strides memory
    struct = NumpyStructure()
    base = addressof(struct)
    offset_shape = addressof(struct.shape_array) - base
    offset_strides = addressof(struct.strides_array) - base

    # Fill the ndarray structure
    struct.nd = len(c_shape)
    struct.data = c_void_p(gpu_data._handle.value)
    struct.dimensions = to_intp_p(base_addr + offset_shape)
    struct.strides = to_intp_p(base_addr + offset_strides)
    struct.shape_array = c_shape
    struct.strides_array = c_strides

    # transfer the memory
    gpu_struct.to_device_raw(addressof(struct), sizeof(struct), stream=stream)

    # NOTE: Do not free gpu_data before freeing gpu_struct.
    gpu_struct.add_dependencies(gpu_data)

    return gpu_struct


def ndarray_data_to_device_memory(ary, stream=0, copy=True, pinned=False):
    dataptr = ary.ctypes.data
    datasize = ndarray_datasize(ary)
    gpu_data = _cuda.DeviceMemory(datasize)

    if copy:
        gpu_data.to_device_raw(dataptr, datasize, stream=stream, pinned=pinned)

    def retriever(stream=0):
        gpu_data.from_device_raw(dataptr, datasize, stream=stream)

    return retriever, gpu_data
