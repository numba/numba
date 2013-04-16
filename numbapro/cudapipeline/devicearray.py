import numpy as np
import warnings
import math
from .ndarray import *
from . import driver as _driver

def is_cuda_ndarray(obj):
    return getattr(obj, '__cuda_ndarray__', False)

def require_cuda_ndarray(obj):
    if not is_cuda_ndarray(obj):
        raise ValueError('require an cuda ndarray object')

def as_arg(ary):
    require_cuda_ndarray(ary)
    return ary.gpu_head

class DeviceNDArray(object):
    __cuda_memory__ = True
    __cuda_ndarray__ = True # There must be a gpu_head and gpu_data attribute
    
    def __init__(self, shape, strides, dtype, stream=0, writeback=None,
                 gpu_head=None, gpu_data=None):
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


def from_array_like(ary, stream=0, gpu_head=None, gpu_data=None):
    return DeviceNDArray(ary.shape, ary.strides, ary.dtype,
                         writeback=ary, stream=stream, gpu_head=gpu_head,
                         gpu_data=gpu_data)



#assert not hasattr(np.ndarray, 'device_allocate')
#assert not hasattr(np.ndarray, 'to_device')
#assert not hasattr(np.ndarray, 'to_host')
#assert not hasattr(np.ndarray, 'free_device')
#assert not hasattr(np.ndarray, 'device_memory')
#assert not hasattr(np.ndarray, 'device_partition')
#assert not hasattr(np.ndarray, 'copy_to_host')
#assert not hasattr(np.ndarray, 'device_mapped')
#
#
#
#class DeviceArrayBase(object):
#    @property
#    def device_memory(self):
#        raise NotImplementedError("Should be overriden in subclass")
#
#    @property
#    def device_raw(self):
#        raise NotImplementedError("Should be overriden in subclass")
#
#    @property
#    def device_raw_ptr(self):
#        return self.device_raw._handle
#
#
#    def copy_to_host(self, array, size=-1, stream=0):
#        if size < 0:
#            size = self.device_raw.bytesize
#        _driver.device_to_host(array, self.device_raw, size, stream=stream)
#
#    def copy_to_device(self, array, stream=0):
#        ndarray_device_transfer_data(array, self.device_raw, stream=stream)
#
#
#class DeviceArray(DeviceArrayBase):
#    '''
#    A memory object that only lives on the device-side.
#    '''
#    def __init__(self, shape, strides, dtype, order, stream=0):
#        ndim = len(shape)
#        self.__shape = shape
#        self.__strides = strides
#        self.__dtype = dtype
#        self.__device_memory = ndarray_device_allocate_struct(ndim)
#        size = _driver.memory_size_from_info(shape, strides, dtype.itemsize)
#        self.__device_data = _driver.DeviceMemory(size)
#        ndarray_populate_struct(self.__device_memory, self.__device_data,
#                                shape, strides, stream=stream)
#
#    @property
#    def device_memory(self):
#        return self.__device_memory
#
#    @property
#    def device_raw(self):
#        return self.__device_data
#
#    @property
#    def shape(self):
#        return self.__shape
#
#    @property
#    def ndim(self):
#        return len(self.shape)
#
#    @property
#    def strides(self):
#        return self.__strides
#
#    @property
#    def size(self):
#        return self.shape[0]
#
#    @property
#    def dtype(self):
#        return self.__dtype
#
#
#class DeviceNDArray(DeviceArrayBase, np.ndarray):
#    @_driver.require_context
#    def device_mapped(self, mappedptr, stream=0):
#        # transfer structure
#        self.__device_memory = ndarray_device_allocate_struct(self.ndim)
#        self.__device_data = mappedptr
#        ndarray_populate_struct(self.__device_memory, self.__device_data,
#                                self.shape, self.strides, stream=stream)
#        self.__gpu_readback = None
#
#    @_driver.require_context
#    def device_allocate(self, stream=0):
#        # do allocation
#        self.__device_memory = ndarray_device_allocate_struct(self.ndim)
#        self.__device_data = ndarray_device_allocate_data(self)
#        ndarray_populate_struct(self.__device_memory, self.__device_data,
#                                self.shape, self.strides, stream=stream)
#        s, e = _driver.host_memory_extents(self)
#        self.__gpu_readback = s, e - s
#
#    @_driver.require_context
#    def to_device(self, stream=0):
#        '''Transfer the ndarray to the device.
#
#        stream --- [optional] The stream to use; or use 0 to imply no stream.
#                              Default to 0.
#        copy --- [optional] Whether to transfer the data or just allocate.
#                            Default to True.
#            '''
#        assert not hasattr(self, '__gpu_readback')
#        ndarray_device_transfer_data(self, self.__device_data, stream=stream)
#
#
#    def to_host(self, stream=0):
#        dataptr, datasize = self.__gpu_readback
#        _driver.device_to_host(dataptr, self.__device_data, datasize,
#                               stream=stream)
#
#    def free_device(self):
#        '''
#        May not always release the device memory immediately.  This only release
#        the reference to the device_memory
#        '''
#        del self.__gpu_readback
#        del self.__device_data
#        del self.__device_memory
#
#    def device_partition(self, idx, stream=0):
#        assert self.ndim == 1
#        assert self.flags['C_CONTIGUOUS']
#        n = self.shape[0]
#        elemsz = self.strides[0]
#        leftn = idx
#        rightn = n - leftn
#        offset = leftn * elemsz
#        
#        left = DeviceNDArray(buffer=self, shape=leftn, dtype=self.dtype)
#        right = DeviceNDArray(buffer=self, offset=offset, shape=rightn,
#                              dtype=self.dtype)
#
#        left.device_allocate(stream=stream)
#        right.device_allocate(stream=stream)
#
#        left.__from_custom_data(self.__device_data, (leftn,), self.strides,
#                                stream=stream)
#
#        offsetted = _driver.DeviceView(self.__device_data, offset)
#        right.__from_custom_data(offsetted, (rightn,), self.strides,
#                                 stream=stream)
#
#        return left, right
#
#    def __from_custom_data(self, data, shape, strides, stream=0):
#        '''
#        Populate the instance with a custom device data memory.
#        '''
#        nd = len(shape)
#        struct = ndarray_device_allocate_struct(nd)
#        ndarray_populate_struct(struct, data, shape, strides)
#        self.__device_memory = struct
#        self.__device_data = data
#
#        assert self.flags['C_CONTIGUOUS']
#        size = shape[0] * strides[0]
#
#        s, e = _driver.host_memory_extents(self)
#        self.__gpu_readback = s, e - s
#        return self
#
#    @property
#    def device_memory(self):
#        try:
#            return self.__device_memory
#        except AttributeError:
#            raise RuntimeError("No GPU device memory for this array")
#
#    @property
#    def device_raw(self):
#        try:
#            return self.__device_data
#        except AttributeError:
#            raise RuntimeError("No GPU device memory for this array")
#
