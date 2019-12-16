from __future__ import print_function, division, absolute_import

import numpy as np

from numba.cuda.testing import unittest, SerialMixin
from numba.cuda.testing import skip_on_cudasim, skip_unless_cudasim
from numba import cuda


class TestCudaArray(SerialMixin, unittest.TestCase):
    def test_gpu_array_zero_length(self):
        x = np.arange(0)
        dx = cuda.to_device(x)
        hx = dx.copy_to_host()
        self.assertEqual(x.shape, dx.shape)
        self.assertEqual(x.size, dx.size)
        self.assertEqual(x.shape, hx.shape)
        self.assertEqual(x.size, hx.size)

    def test_gpu_array_strided(self):

        @cuda.jit('void(double[:])')
        def kernel(x):
            i = cuda.grid(1)
            if i < x.shape[0]:
                x[i] = i

        x = np.arange(10, dtype=np.double)
        y = np.ndarray(shape=10 * 8, buffer=x, dtype=np.byte)
        z = np.ndarray(9, buffer=y[4:-4], dtype=np.double)
        kernel[10, 10](z)
        self.assertTrue(np.allclose(z, list(range(9))))

    def test_gpu_array_interleaved(self):

        @cuda.jit('void(double[:], double[:])')
        def copykernel(x, y):
            i = cuda.grid(1)
            if i < x.shape[0]:
                x[i] = i
                y[i] = i

        x = np.arange(10, dtype=np.double)
        y = x[:-1:2]
        # z = x[1::2]
        # n = y.size
        try:
            cuda.devicearray.auto_device(y)
        except ValueError:
            pass
        else:
            raise AssertionError("Should raise exception complaining the "
                                 "contiguous-ness of the array.")
            # Should we handle this use case?
            # assert z.size == y.size
            # copykernel[1, n](y, x)
            # print(y, z)
            # assert np.all(y == z)
            # assert np.all(y == list(range(n)))

    def test_auto_device_const(self):
        d, _ = cuda.devicearray.auto_device(2)
        self.assertTrue(np.all(d.copy_to_host() == np.array(2)))

    def _test_device_array_like_same(self, d_a):
        """
        Tests of device_array_like where shape, strides, dtype, and flags should
        all be equal.
        """
        d_a_like = cuda.device_array_like(d_a)
        self.assertEqual(d_a.shape, d_a_like.shape)
        self.assertEqual(d_a.strides, d_a_like.strides)
        self.assertEqual(d_a.dtype, d_a_like.dtype)
        self.assertEqual(d_a.flags['C_CONTIGUOUS'], d_a_like.flags['C_CONTIGUOUS'])
        self.assertEqual(d_a.flags['F_CONTIGUOUS'], d_a_like.flags['F_CONTIGUOUS'])

    def test_device_array_like_1d(self):
        d_a = cuda.device_array(10, order='C')
        self._test_device_array_like_same(d_a)

    def test_device_array_like_2d(self):
        d_a = cuda.device_array((10, 12), order='C')
        self._test_device_array_like_same(d_a)

    def test_device_array_like_2d_transpose(self):
        d_a = cuda.device_array((10, 12), order='C')
        self._test_device_array_like_same(d_a.T)

    def test_device_array_like_3d(self):
        d_a = cuda.device_array((10, 12, 14), order='C')
        self._test_device_array_like_same(d_a)

    def test_device_array_like_1d_f(self):
        d_a = cuda.device_array(10, order='F')
        self._test_device_array_like_same(d_a)

    def test_device_array_like_2d_f(self):
        d_a = cuda.device_array((10, 12), order='F')
        self._test_device_array_like_same(d_a)

    def test_device_array_like_2d_f_transpose(self):
        d_a = cuda.device_array((10, 12), order='F')
        self._test_device_array_like_same(d_a.T)

    def test_device_array_like_3d_f(self):
        d_a = cuda.device_array((10, 12, 14), order='F')
        self._test_device_array_like_same(d_a)

    def _test_device_array_like_view(self, view, d_view):
        """
        Tests of device_array_like where the original array is a view - the
        strides should not be equal because a contiguous array is expected.
        """
        d_like = cuda.device_array_like(d_view)
        self.assertEqual(d_view.shape, d_like.shape)
        self.assertEqual(d_view.dtype, d_like.dtype)

        # Use Numpy as a reference for the expected strides
        like = np.zeros_like(view)
        self.assertEqual(d_like.strides, like.strides)
        self.assertEqual(d_like.flags['C_CONTIGUOUS'], like.flags['C_CONTIGUOUS'])
        self.assertEqual(d_like.flags['F_CONTIGUOUS'], like.flags['F_CONTIGUOUS'])

    def test_device_array_like_1d_view(self):
        shape = 10
        view = np.zeros(shape)[::2]
        d_view = cuda.device_array(shape)[::2]
        self._test_device_array_like_view(view, d_view)

    def test_device_array_like_1d_view_f(self):
        shape = 10
        view = np.zeros(shape, order='F')[::2]
        d_view = cuda.device_array(shape, order='F')[::2]
        self._test_device_array_like_view(view, d_view)

    def test_device_array_like_2d_view(self):
        shape = (10, 12)
        view = np.zeros(shape)[::2, ::2]
        d_view = cuda.device_array(shape)[::2, ::2]
        self._test_device_array_like_view(view, d_view)

    def test_device_array_like_2d_view_f(self):
        shape = (10, 12)
        view = np.zeros(shape, order='F')[::2, ::2]
        d_view = cuda.device_array(shape, order='F')[::2, ::2]
        self._test_device_array_like_view(view, d_view)

    @skip_on_cudasim('Numba and numpy stride semantics differ for transpose')
    def test_device_array_like_2d_view_transpose_device(self):
        shape = (10, 12)
        view = np.zeros(shape)[::2, ::2].T
        d_view = cuda.device_array(shape)[::2, ::2].T
        # This is a special case (see issue #4974) because creating the
        # transpose creates a new contiguous allocation with different strides.
        # In this case, rather than comparing against numpy, we can only compare
        # against expected values.
        d_like = cuda.device_array_like(d_view)
        self.assertEqual(d_view.shape, d_like.shape)
        self.assertEqual(d_view.dtype, d_like.dtype)
        self.assertEqual((40, 8), d_like.strides)
        self.assertTrue(d_like.is_c_contiguous())
        self.assertFalse(d_like.is_f_contiguous())

    @skip_unless_cudasim('Numba and numpy stride semantics differ for transpose')
    def test_device_array_like_2d_view_transpose_simulator(self):
        shape = (10, 12)
        view = np.zeros(shape)[::2, ::2].T
        d_view = cuda.device_array(shape)[::2, ::2].T
        # On the simulator, the transpose has different strides to on a CUDA
        # device (See issue #4974). Here we can compare strides against numpy as
        # a reference.
        like = np.zeros_like(view)
        d_like = cuda.device_array_like(d_view)
        self.assertEqual(d_view.shape, d_like.shape)
        self.assertEqual(d_view.dtype, d_like.dtype)
        self.assertEqual(like.strides, d_like.strides)
        self.assertEqual(like.flags['C_CONTIGUOUS'], d_like.flags['C_CONTIGUOUS'])
        self.assertEqual(like.flags['F_CONTIGUOUS'], d_like.flags['F_CONTIGUOUS'])

    def test_device_array_like_2d_view_f_transpose(self):
        shape = (10, 12)
        view = np.zeros(shape, order='F')[::2, ::2].T
        d_view = cuda.device_array(shape, order='F')[::2, ::2].T
        self._test_device_array_like_view(view, d_view)


if __name__ == '__main__':
    unittest.main()
