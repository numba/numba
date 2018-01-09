import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest
from numba.cuda.testing import skip_on_cudasim


@skip_on_cudasim('Device Array API unsupported in the simulator')
class TestCudaNDArray(unittest.TestCase):
    def test_device_array_interface(self):
        dary = cuda.device_array(shape=100)
        devicearray.verify_cuda_ndarray_interface(dary)

        ary = np.empty(100)
        dary = cuda.to_device(ary)
        devicearray.verify_cuda_ndarray_interface(dary)

        ary = np.asarray(1.234)
        dary = cuda.to_device(ary)
        self.assertTrue(dary.ndim == 1)
        devicearray.verify_cuda_ndarray_interface(dary)

    def test_devicearray_no_copy(self):
        array = np.arange(100, dtype=np.float32)
        cuda.to_device(array, copy=False)

    def test_devicearray(self):
        array = np.arange(100, dtype=np.int32)
        original = array.copy()
        gpumem = cuda.to_device(array)
        array[:] = 0
        gpumem.copy_to_host(array)

        self.assertTrue((array == original).all())

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
        self.assertTrue((array == original * 2).all())

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

        self.assertEqual('invalid axes list (0, 0)', str(e.exception))

    def test_devicearray_transpose_wrongaxis(self):
        gpumem = cuda.to_device(np.array(np.arange(12)).reshape(3, 4))

        with self.assertRaises(ValueError) as e:
            np.transpose(gpumem, axes=(0, 2))

        self.assertEqual('invalid axes list (0, 2)', str(e.exception))

    def test_devicearray_transpose_ok(self):
        original = np.array(np.arange(12)).reshape(3, 4)
        array = np.transpose(cuda.to_device(original)).copy_to_host()
        self.assertTrue(np.all(array == original.T))

    def test_devicearray_transpose_T(self):
        original = np.array(np.arange(12)).reshape(3, 4)
        array = cuda.to_device(original).T.copy_to_host()
        self.assertTrue(np.all(array == original.T))


if __name__ == '__main__':
    unittest.main()
