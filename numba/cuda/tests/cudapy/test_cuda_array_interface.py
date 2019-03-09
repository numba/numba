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

    def test_array_views(self):
        """Views created via array interface support:
            - Strided slices
            - Strided slices
        """
        h_arr = np.random.random(10)
        c_arr = cuda.to_device(h_arr)

        arr = cuda.as_cuda_array(c_arr)

        # __getitem__ interface accesses expected data

        # Direct views
        np.testing.assert_array_equal(arr.copy_to_host(), h_arr)
        np.testing.assert_array_equal(arr[:].copy_to_host(), h_arr)

        # Slicing
        np.testing.assert_array_equal(arr[:5].copy_to_host(), h_arr[:5])

        # Strided view
        np.testing.assert_array_equal(arr[::2].copy_to_host(), h_arr[::2])

        # View of strided array
        arr_strided = cuda.as_cuda_array(c_arr[::2])
        np.testing.assert_array_equal(arr_strided.copy_to_host(), h_arr[::2])

        # A strided-view-of-array and view-of-strided-array have the same
        # shape, strides, itemsize, and alloc_size
        self.assertEqual(arr[::2].shape, arr_strided.shape)
        self.assertEqual(arr[::2].strides, arr_strided.strides)
        self.assertEqual(arr[::2].dtype.itemsize, arr_strided.dtype.itemsize)
        self.assertEqual(arr[::2].alloc_size, arr_strided.alloc_size)

        # __setitem__ interface propogates into external array

        # Writes to a slice
        arr[:5] = np.pi
        np.testing.assert_array_equal(
            c_arr.copy_to_host(),
            np.concatenate((np.full(5, np.pi), h_arr[5:]))
        )

        # Writes to a slice from a view
        arr[:5] = arr[5:]
        np.testing.assert_array_equal(
            c_arr.copy_to_host(),
            np.concatenate((h_arr[5:], h_arr[5:]))
        )

        # Writes through a view
        arr[:] = cuda.to_device(h_arr)
        np.testing.assert_array_equal(c_arr.copy_to_host(), h_arr)

        # Writes to a strided slice
        arr[::2] = np.pi
        np.testing.assert_array_equal(
            c_arr.copy_to_host()[::2],
            np.full(5, np.pi),
        )
        np.testing.assert_array_equal(
            c_arr.copy_to_host()[1::2],
            h_arr[1::2]
        )

    def test_negative_strided_issue(self):
        # issue #3705
        h_arr = np.random.random(10)
        c_arr = cuda.to_device(h_arr)

        def base_offset(orig, sliced):
            return sliced['data'][0] - orig['data'][0]

        h_ai = h_arr.__array_interface__
        c_ai = c_arr.__cuda_array_interface__

        h_ai_sliced = h_arr[::-1].__array_interface__
        c_ai_sliced = c_arr[::-1].__cuda_array_interface__

        # Check data offset is correct
        self.assertEqual(
            base_offset(h_ai, h_ai_sliced),
            base_offset(c_ai, c_ai_sliced),
        )
        # Check shape and strides are correct
        self.assertEqual(h_ai_sliced['shape'], c_ai_sliced['shape'])
        self.assertEqual(h_ai_sliced['strides'], c_ai_sliced['strides'])

    def test_negative_strided_copy_to_host(self):
        # issue #3705
        h_arr = np.random.random(10)
        c_arr = cuda.to_device(h_arr)
        sliced = c_arr[::-1]
        with self.assertRaises(NotImplementedError) as raises:
            sliced.copy_to_host()
        expected_msg = 'D->H copy not implemented for negative strides'
        self.assertIn(expected_msg, str(raises.exception))


if __name__ == "__main__":
    unittest.main()
