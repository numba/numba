import numpy as np
from numba import ocl
from numba.ocl.ocldrv import cl
from numba.ocl import oclarray
import unittest
import numpy as np


class TestOCLNDArray(unittest.TestCase):
    def setUp(self):
        self.context = cl.create_context(cl.default_platform,
                                         [cl.default_platform.default_device])

    def tearDown(self):
        del self.context

    def test_device_array_interface(self):
        dary = ocl.device_array(self.context, shape=100)
        oclarray.verify_ocl_ndarray_interface(dary)

        ary = np.empty(100)
        dary = ocl.to_device(self.context, ary)
        oclarray.verify_ocl_ndarray_interface(dary)

        ary = np.asarray(1.234)
        dary = ocl.to_device(self.context, ary)
        self.assertTrue(dary.ndim == 1)
        oclarray.verify_ocl_ndarray_interface(dary)

    def test_devicearray_no_copy(self):
        array = np.arange(100, dtype=np.float32)
        ocl.to_device(self.context, array, copy=False)

    @unittest.skip("todo")
    def test_devicearray(self):
        array = np.arange(100, dtype=np.int32)
        original = array.copy()
        gpumem = cuda.to_device(array)
        array[:] = 0
        gpumem.copy_to_host(array)

        self.assertTrue((array == original).all())

    @unittest.skip("todo")
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

    @unittest.skip("todo")
    def test_devicearray_replace(self):
        N = 100
        array = np.arange(N, dtype=np.int32)
        original = array.copy()
        gpumem = cuda.to_device(array)
        cuda.to_device(array * 2, to=gpumem)
        gpumem.copy_to_host(array)
        self.assertTrue((array == original * 2).all())


if __name__ == '__main__':
    unittest.main()
