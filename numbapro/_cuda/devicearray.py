import numpy as np
from .ndarray import *

assert getattr(np.ndarray, 'synchronous', None) is None

class DeviceNDArray(np.ndarray):
    def to_device(self, stream=0):
        retriever, device_memory = ndarray_to_device_memory(self, stream=stream)
        self.__device_memory = device_memory
        self.__gpu_readback = retriever

    def to_host(self):
        self.__gpu_readback()
        self.free_device()

    def free_device(self):
        del self.__gpu_readback
        del self.__device_memory

    @property
    def device_memory(self):
        try:
            return self.__device_memory
        except AttributeError:
            raise RuntimeError("No GPU device memory for this array")
