import numpy as np
import ctypes
from numba.cuda.cudadrv.devicearray import (DeviceRecord, from_record_like,
                                            auto_device)
from numba import cuda, numpy_support
from numba.cuda.testing import unittest
from numba.cuda.testing import skip_on_cudasim
import numpy as np

@skip_on_cudasim('Device Record API unsupported in the simulator')
class TestCudaDeviceRecord(unittest.TestCase):
    """
    Tests the DeviceRecord class with np.void host types.
    """
    def setUp(self):
        self._create_data(np.zeros)

    def _create_data(self, array_ctor):
        self.dtype = np.dtype([('a', np.int32), ('b', np.float32)], align=True)
        self.hostz = array_ctor(1, self.dtype)[0]
        self.hostnz = array_ctor(1, self.dtype)[0]
        self.hostnz['a'] = 10
        self.hostnz['b'] = 11.0

    def _check_device_record(self, reference, rec):
        self.assertEqual(rec.shape, tuple())
        self.assertEqual(rec.strides, tuple())
        self.assertEqual(rec.dtype, reference.dtype)
        self.assertEqual(rec.alloc_size, reference.dtype.itemsize)
        self.assertIsNotNone(rec.gpu_data)
        self.assertNotEqual(rec.device_ctypes_pointer, ctypes.c_void_p(0))

        numba_type = numpy_support.from_dtype(reference.dtype)
        self.assertEqual(rec._numba_type_, numba_type)

    def test_device_record_interface(self):
        hostrec = self.hostz.copy()
        devrec = DeviceRecord(self.dtype)
        self._check_device_record(hostrec, devrec)

    def test_device_record_copy(self):
        hostrec = self.hostz.copy()
        devrec = DeviceRecord(self.dtype)
        devrec.copy_to_device(hostrec)

        # Copy back and check values are all zeros
        hostrec2 = self.hostnz.copy()
        devrec.copy_to_host(hostrec2)
        np.testing.assert_equal(self.hostz, hostrec2)

        # Copy non-zero values to GPU and back and check values
        hostrec3 = self.hostnz.copy()
        devrec.copy_to_device(hostrec3)

        hostrec4 = self.hostz.copy()
        devrec.copy_to_host(hostrec4)
        np.testing.assert_equal(hostrec4, self.hostnz)

    def test_from_record_like(self):
        # Create record from host record
        hostrec = self.hostz.copy()
        devrec = from_record_like(hostrec)
        self._check_device_record(hostrec, devrec)

        # Create record from device record and check for distinct data
        devrec2 = from_record_like(devrec)
        self._check_device_record(devrec, devrec2)
        self.assertNotEqual(devrec.gpu_data, devrec2.gpu_data)

    def test_auto_device(self):
        # Create record from host record
        hostrec = self.hostnz.copy()
        devrec, new_gpu_obj = auto_device(hostrec)
        self._check_device_record(hostrec, devrec)
        self.assertTrue(new_gpu_obj)

        # Copy data back and check it is equal to auto_device arg
        hostrec2 = self.hostz.copy()
        devrec.copy_to_host(hostrec2)
        np.testing.assert_equal(hostrec2, hostrec)


class TestCudaDeviceRecordWithRecord(TestCudaDeviceRecord):
    """
    Tests the DeviceRecord class with np.record host types
    """
    def setUp(self):
        self._create_data(np.recarray)


if __name__ == '__main__':
    unittest.main()
