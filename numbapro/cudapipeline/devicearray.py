import numpy as np
from .ndarray import *
from . import driver as _driver
from numbapro._utils.ndarray import ndarray_datasize_raw

assert not hasattr(np.ndarray, 'device_allocate')
assert not hasattr(np.ndarray, 'to_device')
assert not hasattr(np.ndarray, 'to_host')
assert not hasattr(np.ndarray, 'free_device')
assert not hasattr(np.ndarray, 'device_memory')
assert not hasattr(np.ndarray, 'device_partition')
assert not hasattr(np.ndarray, 'copy_to_host')
assert not hasattr(np.ndarray, 'device_mapped')

class DeviceArrayBase(object):
    @property
    def device_memory(self):
        raise NotImplementedError("Should be overriden in subclass")

    @property
    def device_raw(self):
        raise NotImplementedError("Should be overriden in subclass")

    @property
    def device_raw_ptr(self):
        return self.device_raw._handle


    def copy_to_host(self, array, size=-1, stream=0):
        if size < 0:
            size = self.device_raw.bytesize
        self.device_raw.from_device_raw(array.ctypes.data, size,
                                           stream=stream)

    def copy_to_device(self, array, stream=0):
        ndarray_device_transfer_data(array, self.device_raw, stream=stream)


class DeviceArray(DeviceArrayBase):
    '''
    A memory object that only lives on the device-side.
    '''
    def __init__(self, shape, strides, dtype, order, stream=0):
        ndim = len(shape)
        self.__shape = shape
        self.__strides = strides
        self.__dtype = dtype
        self.__device_memory = ndarray_device_allocate_struct(ndim)
        size = ndarray_datasize_raw(shape, strides, dtype, order)
        self.__device_data = _driver.AllocatedDeviceMemory(size)
        ndarray_populate_struct(self.__device_memory, self.__device_data,
                                shape, strides, stream=stream)

    @property
    def device_memory(self):
        return self.__device_memory

    @property
    def device_raw(self):
        return self.__device_data

    @property
    def shape(self):
        return self.__shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def strides(self):
        return self.__strides

    @property
    def size(self):
        return self.shape[0]

    @property
    def dtype(self):
        return self.__dtype


class DeviceNDArray(DeviceArrayBase, np.ndarray):
    @_driver.require_context
    def device_mapped(self, mappedptr, stream=0):
        # transfer structure
        self.__device_memory = ndarray_device_allocate_struct(self.ndim)
        self.__device_data = mappedptr
        ndarray_populate_struct(self.__device_memory, self.__device_data,
                                self.shape, self.strides, stream=stream)
        self.__gpu_readback = None

    @_driver.require_context
    def device_allocate(self, stream=0):
        # do allocation
        self.__device_memory = ndarray_device_allocate_struct(self.ndim)
        self.__device_data = ndarray_device_allocate_data(self)
        ndarray_populate_struct(self.__device_memory, self.__device_data,
                                self.shape, self.strides, stream=stream)
        self.__gpu_readback = self.ctypes.data, ndarray_datasize(self)

    @_driver.require_context
    def to_device(self, stream=0):
        '''Transfer the ndarray to the device.

        stream --- [optional] The stream to use; or use 0 to imply no stream.
                              Default to 0.
        copy --- [optional] Whether to transfer the data or just allocate.
                            Default to True.
            '''
        assert not hasattr(self, '__gpu_readback')
        ndarray_device_transfer_data(self, self.__device_data, stream=stream)


    def to_host(self, stream=0):
        dataptr, datasize = self.__gpu_readback
        self.__device_data.from_device_raw(dataptr, datasize, stream=stream)

    def free_device(self):
        '''
        May not always release the device memory immediately.  This only release
        the reference to the device_memory
        '''
        del self.__gpu_readback
        del self.__device_data
        del self.__device_memory

    def device_partition(self, idx, stream=0):
        assert self.ndim == 1
        assert self.flags['C_CONTIGUOUS']
        n = self.shape[0]
        elemsz = self.strides[0]
        leftn = idx
        rightn = n - leftn
        offset = leftn * elemsz
        
        left = DeviceNDArray(buffer=self, shape=leftn, dtype=self.dtype)
        right = DeviceNDArray(buffer=self, offset=offset, shape=rightn,
                              dtype=self.dtype)

        left.device_allocate(stream=stream)
        right.device_allocate(stream=stream)

        left.__from_custom_data(self.__device_data, (leftn,), self.strides,
                                stream=stream)

        offsetted = self.__device_data.offset(offset)
        right.__from_custom_data(offsetted, (rightn,), self.strides,
                                 stream=stream)

        return left, right

    def __from_custom_data(self, data, shape, strides, stream=0):
        '''
        Populate the instance with a custom device data memory.
        '''
        nd = len(shape)
        struct = ndarray_device_allocate_struct(nd)
        ndarray_populate_struct(struct, data, shape, strides)
        self.__device_memory = struct
        self.__device_data = data

        assert self.flags['C_CONTIGUOUS']
        size = shape[0] * strides[0]

        self.__gpu_readback = self.ctypes.data, size
        return self

    @property
    def device_memory(self):
        try:
            return self.__device_memory
        except AttributeError:
            raise RuntimeError("No GPU device memory for this array")

    @property
    def device_raw(self):
        try:
            return self.__device_data
        except AttributeError:
            raise RuntimeError("No GPU device memory for this array")

