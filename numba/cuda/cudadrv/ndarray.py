from __future__ import print_function, absolute_import, division
import numba.ctypes_support as ctypes
from . import devices, driver


def make_array_ctype(ndim):
    c_intp = ctypes.c_ssize_t

    class c_array(ctypes.Structure):
        _fields_ = [('data', ctypes.c_void_p),
                    ('shape', c_intp * ndim),
                    ('strides', c_intp * ndim)]

    return c_array


def ndarray_device_allocate_head(nd):
    "Allocate the metadata structure"
    arraytype = make_array_ctype(nd)
    gpu_head = devices.get_context().memalloc(ctypes.sizeof(arraytype))
    return gpu_head


def ndarray_device_allocate_data(ary):
    datasize = driver.host_memory_size(ary)
    # allocate
    gpu_data = devices.get_context().memalloc(datasize)
    return gpu_data


def ndarray_device_transfer_data(ary, gpu_data, stream=0):
    size = driver.host_memory_size(ary)
    # transfer data
    driver.host_to_device(gpu_data, ary, size, stream=stream)


def ndarray_populate_head(gpu_head, gpu_data, shape, strides, stream=0):
    nd = len(shape)
    assert nd > 0, "0 or negative dimension"

    arraytype = make_array_ctype(nd)
    struct = arraytype(data=driver.device_pointer(gpu_data),
                       shape=shape,
                       strides=strides)

    driver.host_to_device(gpu_head, struct, ctypes.sizeof(struct),
                          stream=stream)
    driver.device_memory_depends(gpu_head, gpu_data)
