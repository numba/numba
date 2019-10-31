import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest, SerialMixin
from numba.cuda.testing import skip_on_cudasim


class TestCudaNDArray(SerialMixin, unittest.TestCase):
    def test_device_array_interface(self):
        dary = cuda.device_array(shape=100)
        devicearray.verify_cuda_ndarray_interface(dary)

        ary = np.empty(100)
        dary = cuda.to_device(ary)
        devicearray.verify_cuda_ndarray_interface(dary)

        ary = np.asarray(1.234)
        dary = cuda.to_device(ary)
        self.assertEquals(dary.ndim, 1)
        devicearray.verify_cuda_ndarray_interface(dary)

    def test_device_array_from_readonly(self):
        ary = np.arange(100, dtype=np.float32)
        # Make the array readonly
        ary.flags.writeable = False
        self.assertFalse(ary.flags.writeable)
        # Ensure that we can copy the readonly array
        dary = cuda.to_device(ary)
        retr = dary.copy_to_host()
        np.testing.assert_array_equal(retr, ary)

    def test_devicearray_no_copy(self):
        array = np.arange(100, dtype=np.float32)
        cuda.to_device(array, copy=False)

    def test_devicearray_shape(self):
        ary = np.arange(2 * 3 * 4).reshape(2, 3, 4)
        dary = cuda.to_device(ary)
        self.assertEquals(ary.shape, dary.shape)
        self.assertEquals(ary.shape[1:], dary.shape[1:])

    def test_devicearray(self):
        array = np.arange(100, dtype=np.int32)
        original = array.copy()
        gpumem = cuda.to_device(array)
        array[:] = 0
        gpumem.copy_to_host(array)

        np.testing.assert_array_equal(array, original)

    def test_stream_bind(self):
        stream = cuda.stream()
        with stream.auto_synchronize():
            arr = cuda.device_array(
                (3, 3),
                dtype=np.float64,
                stream=stream)
            self.assertEqual(arr.bind(stream).stream, stream)
            self.assertEqual(arr.stream, stream)

    def test_len_1d(self):
        ary = np.empty((3,))
        dary = cuda.device_array(3)
        self.assertEqual(len(ary), len(dary))

    def test_len_2d(self):
        ary = np.empty((3, 5))
        dary = cuda.device_array((3, 5))
        self.assertEqual(len(ary), len(dary))

    def test_len_3d(self):
        ary = np.empty((3, 5, 7))
        dary = cuda.device_array((3, 5, 7))
        self.assertEqual(len(ary), len(dary))

    def test_devicearray_partition(self):
        N = 100
        array = np.arange(N, dtype=np.int32)
        original = array.copy()
        gpumem = cuda.to_device(array)
        left, right = gpumem.split(N // 2)

        array[:] = 0

        self.assertTrue(np.all(array == 0))

        right.copy_to_host(array[N//2:])
        left.copy_to_host(array[:N//2])

        self.assertTrue(np.all(array == original))

    def test_devicearray_replace(self):
        N = 100
        array = np.arange(N, dtype=np.int32)
        original = array.copy()
        gpumem = cuda.to_device(array)
        cuda.to_device(array * 2, to=gpumem)
        gpumem.copy_to_host(array)
        np.testing.assert_array_equal(array, original * 2)

    @skip_on_cudasim('This works in the simulator')
    def test_devicearray_transpose_wrongdim(self):
        gpumem = cuda.to_device(np.array(np.arange(12)).reshape(3, 4, 1))

        with self.assertRaises(NotImplementedError) as e:
            np.transpose(gpumem)

        self.assertEqual(
            "transposing a non-2D DeviceNDArray isn't supported",
            str(e.exception))

    def test_devicearray_transpose_identity(self):
        # any-shape identities should work
        original = np.array(np.arange(24)).reshape(3, 4, 2)
        array = np.transpose(cuda.to_device(original), axes=(0, 1, 2)).copy_to_host()
        self.assertTrue(np.all(array == original))

    def test_devicearray_transpose_duplicatedaxis(self):
        gpumem = cuda.to_device(np.array(np.arange(12)).reshape(3, 4))

        with self.assertRaises(ValueError) as e:
            np.transpose(gpumem, axes=(0, 0))

        self.assertIn(
            str(e.exception),
            container=[
                'invalid axes list (0, 0)',  # GPU
                'repeated axis in transpose',  # sim
            ])

    def test_devicearray_transpose_wrongaxis(self):
        gpumem = cuda.to_device(np.array(np.arange(12)).reshape(3, 4))

        with self.assertRaises(ValueError) as e:
            np.transpose(gpumem, axes=(0, 2))

        self.assertIn(
            str(e.exception),
            container=[
                'invalid axes list (0, 2)',  # GPU
                'invalid axis for this array',
                'axis 2 is out of bounds for array of dimension 2',  # sim
            ])

    def test_devicearray_transpose_ok(self):
        original = np.array(np.arange(12)).reshape(3, 4)
        array = np.transpose(cuda.to_device(original)).copy_to_host()
        self.assertTrue(np.all(array == original.T))

    def test_devicearray_transpose_T(self):
        original = np.array(np.arange(12)).reshape(3, 4)
        array = cuda.to_device(original).T.copy_to_host()
        self.assertTrue(np.all(array == original.T))

    def test_devicearray_contiguous_slice(self):
        # memcpys are dumb ranges of bytes, so trying to
        # copy to a non-contiguous range shouldn't work!
        a = np.arange(25).reshape(5, 5, order='F')
        s = np.full(fill_value=5, shape=(5,))

        d = cuda.to_device(a)
        a[2] = s

        # d is in F-order (not C-order), so d[2] is not contiguous
        # (40-byte strides). This means we can't memcpy to it!
        with self.assertRaises(ValueError) as e:
            d[2].copy_to_device(s)
        self.assertEqual(
            devicearray.errmsg_contiguous_buffer,
            str(e.exception))

        # if d[2].copy_to_device(s), then this would pass:
        # self.assertTrue((a == d.copy_to_host()).all())

    def _test_devicearray_contiguous_host_copy(self, a_c, a_f):
        """
        Checks host->device memcpys
        """
        self.assertTrue(a_c.flags.c_contiguous)
        self.assertTrue(a_f.flags.f_contiguous)

        for original, copy in [
            (a_f, a_f),
            (a_f, a_c),
            (a_c, a_f),
            (a_c, a_c),
        ]:
            msg = '%s => %s' % (
                'C' if original.flags.c_contiguous else 'F',
                'C' if copy.flags.c_contiguous else 'F',
            )

            d = cuda.to_device(original)
            d.copy_to_device(copy)
            self.assertTrue(np.all(d.copy_to_host() == a_c), msg=msg)
            self.assertTrue(np.all(d.copy_to_host() == a_f), msg=msg)

    def test_devicearray_contiguous_copy_host_3d(self):
        a_c = np.arange(5 * 5 * 5).reshape(5, 5, 5)
        a_f = np.array(a_c, order='F')
        self._test_devicearray_contiguous_host_copy(a_c, a_f)

    def test_devicearray_contiguous_copy_host_1d(self):
        a_c = np.arange(5)
        a_f = np.array(a_c, order='F')
        self._test_devicearray_contiguous_host_copy(a_c, a_f)

    def test_devicearray_contiguous_copy_device(self):
        a_c = np.arange(5 * 5 * 5).reshape(5, 5, 5)
        a_f = np.array(a_c, order='F')
        self.assertTrue(a_c.flags.c_contiguous)
        self.assertTrue(a_f.flags.f_contiguous)

        d = cuda.to_device(a_c)

        with self.assertRaises(ValueError) as e:
            d.copy_to_device(cuda.to_device(a_f))
        self.assertEqual(
            "incompatible strides: {} vs. {}".format(a_c.strides, a_f.strides),
            str(e.exception))

        d.copy_to_device(cuda.to_device(a_c))
        self.assertTrue(np.all(d.copy_to_host() == a_c))

        d = cuda.to_device(a_f)

        with self.assertRaises(ValueError) as e:
            d.copy_to_device(cuda.to_device(a_c))
        self.assertEqual(
            "incompatible strides: {} vs. {}".format(a_f.strides, a_c.strides),
            str(e.exception))

        d.copy_to_device(cuda.to_device(a_f))
        self.assertTrue(np.all(d.copy_to_host() == a_f))

    def test_devicearray_broadcast_host_copy(self):
        try:
            broadcast_to = np.broadcast_to
        except AttributeError:
            # numpy<1.10 doesn't have broadcast_to. The following implements
            # a limited broadcast_to that only works along already existing
            # dimensions of length 1.
            def broadcast_to(arr, new_shape):
                new_strides = []
                for new_length, length, stride in zip(
                    new_shape, arr.shape, arr.strides
                ):
                    if length == 1 and new_length > 1:
                        new_strides.append(0)
                    elif new_length == length:
                        new_strides.append(stride)
                    else:
                        raise ValueError(
                            "cannot broadcast shape {} to shape {}"
                            .format(arr.shape, new_shape)
                        )
                return np.ndarray(
                    buffer=np.squeeze(arr),
                    dtype=arr.dtype,
                    shape=new_shape,
                    strides=tuple(new_strides),
                )

        broadsize = 4
        coreshape = (2, 3)
        coresize = np.prod(coreshape)
        core_c = np.arange(coresize).reshape(coreshape, order='C')
        core_f = np.arange(coresize).reshape(coreshape, order='F')
        for dim in range(len(coreshape)):
            newindex = (slice(None),) * dim + (np.newaxis,)
            broadshape = coreshape[:dim] + (broadsize,) + coreshape[dim:]
            broad_c = broadcast_to(core_c[newindex], broadshape)
            broad_f = broadcast_to(core_f[newindex], broadshape)
            dbroad_c = cuda.to_device(broad_c)
            dbroad_f = cuda.to_device(broad_f)
            np.testing.assert_array_equal(dbroad_c.copy_to_host(), broad_c)
            np.testing.assert_array_equal(dbroad_f.copy_to_host(), broad_f)
            # Also test copying across different core orderings
            dbroad_c.copy_to_device(broad_f)
            dbroad_f.copy_to_device(broad_c)
            np.testing.assert_array_equal(dbroad_c.copy_to_host(), broad_f)
            np.testing.assert_array_equal(dbroad_f.copy_to_host(), broad_c)

    def test_devicearray_contiguous_host_strided(self):
        a_c = np.arange(10)
        d = cuda.to_device(a_c)
        arr = np.arange(20)[::2]
        d.copy_to_device(arr)
        np.testing.assert_array_equal(d.copy_to_host(), arr)

    def test_devicearray_contiguous_device_strided(self):
        d = cuda.to_device(np.arange(20))
        arr = np.arange(20)

        with self.assertRaises(ValueError) as e:
            d.copy_to_device(cuda.to_device(arr)[::2])
        self.assertEqual(
            devicearray.errmsg_contiguous_buffer,
            str(e.exception))


class TestRecarray(SerialMixin, unittest.TestCase):
    def test_recarray(self):
        # From issue #4111
        a = np.recarray((16,), dtype=[
            ("value1", np.int64),
            ("value2", np.float64),
        ])
        a.value1 = np.arange(a.size, dtype=np.int64)
        a.value2 = np.arange(a.size, dtype=np.float64) / 100

        expect1 = a.value1
        expect2 = a.value2

        def test(x, out1, out2):
            i = cuda.grid(1)
            if i < x.size:
                out1[i] = x.value1[i]
                out2[i] = x.value2[i]

        got1 = np.zeros_like(expect1)
        got2 = np.zeros_like(expect2)
        cuda.jit(test)[1, a.size](a, got1, got2)

        np.testing.assert_array_equal(expect1, got1)
        np.testing.assert_array_equal(expect2, got2)


if __name__ == '__main__':
    unittest.main()
