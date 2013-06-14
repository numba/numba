'''

A CUDA ND Array is recognized by checking the __cuda_memory__ attribute
on the object.  If it exists and evaluate to True, it must define shape,
strides, dtype and size attributes similar to a NumPy ndarray.
'''

import numpy as np
import warnings
import math
from .ndarray import *
from . import driver as _driver

def is_cuda_ndarray(obj):
    "Check if an object is a CUDA ndarray"
    return getattr(obj, '__cuda_ndarray__', False)

def verify_cuda_ndarray_interface(obj):
    "Verify the CUDA ndarray interface for an obj"
    require_cuda_ndarray(obj)
    def requires_attr(attr, typ):
        if not hasattr(obj, attr):
            raise AttributeError(attr)
        if not isinstance(getattr(obj, attr), typ):
            raise AttributeError('%s must be of type %s' % (attr, typ))
    requires_attr('shape', tuple)
    requires_attr('strides', tuple)
    requires_attr('dtype', np.dtype)
    requires_attr('size', (int, long))

def require_cuda_ndarray(obj):
    "Raises ValueError is is_cuda_ndarray(obj) evaluates False"
    if not is_cuda_ndarray(obj):
        raise ValueError('require an cuda ndarray object')

class DeviceNDArray(object):
    '''A on GPU NDArray representation
    '''
    __cuda_memory__ = True
    __cuda_ndarray__ = True # There must be a gpu_head and gpu_data attribute
    
    def __init__(self, shape, strides, dtype, stream=0, writeback=None,
                 gpu_head=None, gpu_data=None):
        '''
        Arguments

        shape: array shape.
        strides: array strides.
        dtype: data type as numpy.dtype.
        stream: cuda stream.
        writeback: Deprecated.
        gpu_head: user provided device memory for the ndarray head structure
        gpu_data: user provided device memory for the ndarray data buffer
        '''
        if isinstance(shape, (int, long)):
            shape = (shape,)
        if isinstance(strides, (int, long)):
            strides = (strides,)
        self.ndim = len(shape)
        if len(strides) != self.ndim:
            raise ValueError('strides not match ndim')
        self.shape = tuple(shape)
        self.strides = tuple(strides)
        self.dtype = np.dtype(dtype)
        self.size = np.prod(self.shape)
        # prepare gpu memory
        if gpu_data is None:
            self.alloc_size = _driver.memory_size_from_info(self.shape,
                                                            self.strides,
                                                            self.dtype.itemsize)
            gpu_data = _driver.DeviceMemory(self.alloc_size)
        else:
            self.alloc_size = _driver.device_memory_size(gpu_data)

        if gpu_head is None:
            gpu_head = ndarray_device_allocate_head(self.ndim)

            ndarray_populate_head(gpu_head, gpu_data, self.shape,
                                  self.strides, stream=stream)
        self.gpu_head = gpu_head
        self.gpu_data = gpu_data

        self.__writeback = writeback # should deprecate the use of this

    @property
    def device_ctypes_pointer(self):
        "Returns the ctypes pointer to the GPU data buffer"
        return self.gpu_data.device_ctypes_pointer

    def copy_to_device(self, ary, stream=0):
        """Copy `ary` to `self`.
        
        If `ary` is a CUDA memory, perform a device-to-device transfer.
        Otherwise, perform a a host-to-device transfer.
        """
        if _driver.is_device_memory(ary):
            sz = min(_driver.device_memory_size(self),
                     _driver.device_memory_size(ary))
            _driver.device_to_device(self, ary, sz, stream=stream)
        else:
            sz = min(_driver.host_memory_size(ary), self.alloc_size)
            _driver.host_to_device(self, ary, sz, stream=stream)

    def copy_to_host(self, ary=None, stream=0):
        """Copy `self` to `ary` or create an numpy ndarray is ary is None.
        """
        if ary is None:
            ary = np.empty(shape=self.shape, dtype=self.dtype)
        _driver.device_to_host(ary, self, self.alloc_size, stream=stream)
        return ary

    def to_host(self, stream=0):
        warnings.warn("to_host() is deprecated and will be removed",
                      DeprecationWarning)
        if self.__writeback is None:
            raise ValueError("no associated writeback array")
        self.copy_to_host(self.__writeback, stream=stream)

    def split(self, section, stream=0):
        '''Split the array into equal partition of the `section` size.
        If the array cannot be equally divided, the last section will be 
        smaller.
        '''
        if self.ndim != 1:
            raise ValueError("only support 1d array")
        if self.strides[0] != self.dtype.itemsize:
            raise ValueError("only support unit stride")
        nsect = int(math.ceil(float(self.size) / section))
        strides = self.strides
        itemsize = self.dtype.itemsize
        for i in range(nsect):
            begin = i * section
            end = min(begin + section, self.size)
            shape = (end - begin,)
            gpu_data = _driver.DeviceView(self.gpu_data, begin * itemsize,
                                          end * itemsize)
            gpu_head = ndarray_device_allocate_head(1)
            ndarray_populate_head(gpu_head, gpu_data, shape, strides,
                                  stream=stream)
            yield DeviceNDArray(shape, strides, dtype=self.dtype, stream=stream,
                                gpu_head=gpu_head, gpu_data=gpu_data)

    def as_cuda_arg(self):
        '''Returns a device memory object that is used as the argument.
        '''
        return self.gpu_head


class MappedNDArray(DeviceNDArray, np.ndarray):
    def device_setup(self, gpu_data, stream=0):
        gpu_head = ndarray_device_allocate_head(self.ndim)

        ndarray_populate_head(gpu_head, gpu_data, self.shape,
                              self.strides, stream=stream)

        self.gpu_data = gpu_data
        self.gpu_head = gpu_head

def from_array_like(ary, stream=0, gpu_head=None, gpu_data=None):
    "Create a DeviceNDArray object that is like ary."
    if ary.ndim == 0:
        ary = ary.reshape(1)
    return DeviceNDArray(ary.shape, ary.strides, ary.dtype,
                         writeback=ary, stream=stream, gpu_head=gpu_head,
                         gpu_data=gpu_data)

def auto_device(ary, stream=0, copy=True):
    if _driver.is_device_memory(ary):
        return ary, False
    else:
        devarray = from_array_like(ary, stream=stream)
        if copy:
            devarray.copy_to_device(ary, stream=stream)
        return devarray, True
