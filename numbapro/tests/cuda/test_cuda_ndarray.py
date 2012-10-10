import numpy as np

from numbapro._cuda.driver import *
from numbapro._cuda.default import *
from numbapro._cuda.ndarray import *
from ctypes import *


array = np.arange(100, dtype=np.float32)
original = array.copy()
retriever, gpu_struct = ndarray_to_device_memory(array)
array[:] = 0
retriever()

assert (array == original).all()

