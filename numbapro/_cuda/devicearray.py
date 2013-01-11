import numpy as np
from .ndarray import *

assert not hasattr(np.ndarray, 'device_allocate')
assert not hasattr(np.ndarray, 'to_device')
assert not hasattr(np.ndarray, 'to_host')
assert not hasattr(np.ndarray, 'free_device')
assert not hasattr(np.ndarray, 'device_memory')
assert not hasattr(np.ndarray, 'device_partition')
assert not hasattr(np.ndarray, 'copy_to_host')
assert not hasattr(np.ndarray, 'device_mapped')


class DeviceNDArray(np.ndarray):
    def device_mapped(self, mappedptr, stream=0):
        # Ensure we already have a CUDA device
        from . import default # this creates the default context if none exist
        # transfer structure
        self.__device_memory = ndarray_device_allocate_struct(self.ndim)
        self.__device_data = mappedptr
        ndarray_populate_struct(self.__device_memory, self.__device_data,
                                self.ctypes.shape, self.ctypes.strides,
                                stream=stream)
        self.__gpu_readback = None

    def device_allocate(self, stream=0):
        # Ensure we already have a CUDA device
        from . import default # this creates the default context if none exist
        # do allocation
        self.__device_memory = ndarray_device_allocate_struct(self.ndim)
        self.__device_data = ndarray_device_allocate_data(self)
        ndarray_populate_struct(self.__device_memory, self.__device_data,
                                self.ctypes.shape, self.ctypes.strides,
                                stream=stream)
        self.__gpu_readback = self.ctypes.data, ndarray_datasize(self)

    def to_device(self, stream=0):
        '''Transfer the ndarray to the device.

        stream --- [optional] The stream to use; or use 0 to imply no stream.
                              Default to 0.
        copy --- [optional] Whether to transfer the data or just allocate.
                            Default to True.
            '''
        from . import default # this creates the default context if none exist
        assert not hasattr(self, '__gpu_readback')
        ndarray_device_transfer_data(self, self.__device_data, stream=stream)


    def to_host(self, stream=0):
        dataptr, datasize = self.__gpu_readback
        self.__device_data.from_device_raw(dataptr, datasize, stream=stream)

    def copy_to_host(self, array, size, stream=0):
        self.__device_data.from_device_raw(array.ctypes.data, size,
                                           stream=stream)

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
        c_shape = (np.ctypeslib.c_intp * len(shape))(*shape)
        c_strides = (np.ctypeslib.c_intp * len(strides))(*strides)
        struct = ndarray_device_allocate_struct(len(c_shape))
        ndarray_populate_struct(struct, data, c_shape, c_strides)
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
