import numpy as np
from .ndarray import *

assert not hasattr(np.ndarray, 'to_device')
assert not hasattr(np.ndarray, 'to_host')
assert not hasattr(np.ndarray, 'free_device')
assert not hasattr(np.ndarray, 'device_memory')
assert not hasattr(np.ndarray, 'device_partition')

class DeviceNDArray(np.ndarray):
    def to_device(self, stream=0, copy=True):
        '''Transfer the ndarray to the device.

        '''
        assert not hasattr(self, '__device_memory')
        assert not hasattr(self, '__gpu_readback')
        packed = ndarray_device_memory_and_data(self, stream=stream, copy=copy)
        retriever, device_memory, device_data = packed
        self.__device_memory = device_memory
        self.__device_data = device_data
        self.__gpu_readback = retriever

    def to_host(self, stream=0):
        self.__gpu_readback(stream=stream)

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
        n = self.shape[0]
        elemsz = self.strides[0]
        halfn = n / 2
        halfsz = halfn * elemsz

        left = DeviceNDArray(buffer=self, shape=halfn, dtype=self.dtype)
        right = DeviceNDArray(buffer=self, offset=halfsz, shape=halfn,
                              dtype=self.dtype)

        left.__from_custom_data(self.__device_data, (halfn,), self.strides,
                                stream=stream)

        offsetted = self.__device_data.offset(halfsz)
        right.__from_custom_data(offsetted, (halfn,), self.strides,
                                 stream=stream)

        # add dependencies to prevent free the base data pointer
        left.__device_data.add_dependencies(self.__device_data)
        right.__device_data.add_dependencies(self.__device_data)

        return left, right

    def __from_custom_data(self, data, shape, strides, stream=0):
        '''
        Populate the instance with a custom device data memory.
        '''
        c_shape = (np.ctypeslib.c_intp * len(shape))(*shape)
        c_strides = (np.ctypeslib.c_intp * len(strides))(*strides)
        struct = ndarray_device_memory_from_data(data, c_shape, c_strides,
                                                 stream=stream)
        self.__device_memory = struct
        self.__device_data = data

        size = shape[0] * strides[0]
        def readback(stream=0):
            self.__device_data.from_device_raw(self.ctypes.data, size,
                                               stream=stream)

        self.__gpu_readback = readback
        return self


    @property
    def device_memory(self):
        try:
            return self.__device_memory
        except AttributeError:
            raise RuntimeError("No GPU device memory for this array")
