import numpy as np

from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim


class MyArray(object):
    def __init__(self, arr):
        self._arr = arr
        self.__cuda_array_interface__ = arr.__cuda_array_interface__


@skip_on_cudasim('CUDA Memory API unsupported in the simulator')
class TestCudaArrayInterface(CUDATestCase):
    def test_as_cuda_array(self):
        h_arr = np.arange(10)
        d_arr = cuda.to_device(h_arr)
        my_arr = MyArray(d_arr)
        wrapped = cuda.as_cuda_array(my_arr)

        np.testing.assert_array_equal(wrapped.copy_to_host(), h_arr)
        np.testing.assert_array_equal(d_arr.copy_to_host(), h_arr)

    def test_kernel_arg(self):
        h_arr = np.arange(10)
        d_arr = cuda.to_device(h_arr)
        my_arr = MyArray(d_arr)
        wrapped = cuda.as_cuda_array(my_arr)

        @cuda.jit
        def mutate(arr, val):
            arr[cuda.grid(1)] += val

        val = 7
        mutate.forall(wrapped.size)(wrapped, val)

        np.testing.assert_array_equal(wrapped.copy_to_host(), h_arr + val)
        np.testing.assert_array_equal(d_arr.copy_to_host(), h_arr + val)


if __name__ == '__main__':
    unittest.main()
