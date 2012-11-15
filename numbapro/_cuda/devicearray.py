import numpy as np
from .ndarray import *

assert not hasattr(np.ndarray, 'to_device')
assert not hasattr(np.ndarray, 'to_host')
assert not hasattr(np.ndarray, 'free_device')
assert not hasattr(np.ndarray, 'device_memory')

class DeviceNDArray(np.ndarray):
    def to_device(self, stream=0, copy=True):
        assert not hasattr(self, '__device_memory')
        assert not hasattr(self, '__gpu_readback')
        retriever, device_memory = ndarray_to_device_memory(self,
                                                            stream=stream,
                                                            copy=copy)
        self.__device_memory = device_memory
        self.__gpu_readback = retriever

    def to_host(self, stream=0):
        self.__gpu_readback(stream=stream)

    def free_device(self):
        del self.__gpu_readback
        del self.__device_memory

    @property
    def device_memory(self):
        try:
            return self.__device_memory
        except AttributeError:
            raise RuntimeError("No GPU device memory for this array")

