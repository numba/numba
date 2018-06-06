import numpy as np

from numba import vectorize, guvectorize
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim


class MyArray(object):
    def __init__(self, arr):
        self._arr = arr
        self.__cuda_array_interface__ = arr.__cuda_array_interface__


@skip_on_cudasim('CUDA Array Interface is not supported in the simulator')
class TestCudaArrayInterface(CUDATestCase):
    def test_as_cuda_array(self):
        h_arr = np.arange(10)
        self.assertFalse(cuda.is_cuda_array(h_arr))
        d_arr = cuda.to_device(h_arr)
        self.assertTrue(cuda.is_cuda_array(d_arr))
        my_arr = MyArray(d_arr)
        self.assertTrue(cuda.is_cuda_array(my_arr))
        wrapped = cuda.as_cuda_array(my_arr)
        self.assertTrue(cuda.is_cuda_array(wrapped))
        # Their values must equal the original array
        np.testing.assert_array_equal(wrapped.copy_to_host(), h_arr)
        np.testing.assert_array_equal(d_arr.copy_to_host(), h_arr)
        # d_arr and wrapped must be the same buffer
        self.assertEqual(wrapped.device_ctypes_pointer.value,
                         d_arr.device_ctypes_pointer.value)

    def test_ownership(self):
        # Get the deallocation queue
        ctx = cuda.current_context()
        deallocs = ctx.deallocations
        # Flush all deallocations
        deallocs.clear()
        self.assertEqual(len(deallocs), 0)
        # Make new device array
        d_arr = cuda.to_device(np.arange(100))
        # Convert it
        cvted = cuda.as_cuda_array(d_arr)
        # Drop reference to the original object such that
        # only `cvted` has a reference to it.
        del d_arr
        # There shouldn't be any new deallocations
        self.assertEqual(len(deallocs), 0)
        # Try to access the memory and verify its content
        np.testing.assert_equal(cvted.copy_to_host(), np.arange(100))
        # Drop last reference to the memory
        del cvted
        self.assertEqual(len(deallocs), 1)
        # Flush
        deallocs.clear()

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

    def test_ufunc_arg(self):
        @vectorize(['f8(f8, f8)'], target='cuda')
        def vadd(a, b):
            return a + b

        # Case 1: use custom array as argument
        h_arr = np.random.random(10)
        arr = MyArray(cuda.to_device(h_arr))
        val = 6
        out = vadd(arr, val)
        np.testing.assert_array_equal(out.copy_to_host(), h_arr + val)

        # Case 2: use custom array as return
        out = MyArray(cuda.device_array(h_arr.shape))
        returned = vadd(h_arr, val, out=out)
        np.testing.assert_array_equal(returned.copy_to_host(), h_arr + val)

    def test_gufunc_arg(self):
        @guvectorize(['(f8, f8, f8[:])'], '(),()->()', target='cuda')
        def vadd(inp, val, out):
            out[0] = inp + val

        # Case 1: use custom array as argument
        h_arr = np.random.random(10)
        arr = MyArray(cuda.to_device(h_arr))
        val = np.float64(7)
        out = vadd(arr, val)
        np.testing.assert_array_equal(out.copy_to_host(), h_arr + val)

        # Case 2: use custom array as return
        out = MyArray(cuda.device_array(h_arr.shape))
        returned = vadd(h_arr, val, out=out)
        np.testing.assert_array_equal(returned.copy_to_host(), h_arr + val)
        self.assertEqual(returned.device_ctypes_pointer.value,
                         out._arr.device_ctypes_pointer.value)


if __name__ == '__main__':
    unittest.main()
