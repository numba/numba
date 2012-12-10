import numpy as np
import unittest 
from numbapro._cuda.driver import *
from numbapro._cuda.default import *
from numbapro._cuda.ndarray import *
from numbapro import cuda
from ctypes import *

import support

class TestCudaNDArray(support.CudaTestCase):
    def test_ndarray(self):
        array = np.arange(100, dtype=np.float32)
        original = array.copy()
        retriever, gpu_struct = ndarray_to_device_memory(array)
        array[:] = 0
        retriever()

        self.assertTrue((array == original).all())

    def test_devicearray(self):
        array = np.arange(100, dtype=np.int32)
        original = array.copy()
        gpumem = cuda.to_device(array)
        array[:] = 0
        gpumem.to_host()

        self.assertTrue((array == original).all())

    def test_devicearray_partition(self):
        N = 100
        array = np.arange(N, dtype=np.int32)
        original = array.copy()
        gpumem = cuda.to_device(array)
        left, right = gpumem.device_partition(N / 2)
        array[:] = 0

        self.assertTrue(left.sum()==0)
        self.assertTrue(right.sum()==0)

        left.to_host()
        self.assertTrue((left == original[:N/2]).all())
        self.assertTrue(right.sum()==0)


        right.to_host()
        self.assertTrue((left == original[:N/2]).all())
        self.assertTrue((right == original[N/2:]).all())

        self.assertTrue((array == original).all())


if __name__ == '__main__':
    unittest.main()