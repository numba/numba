import unittest
import numpy
from numbapro.cudapipeline import driver

class TestCudaMemory(unittest.TestCase):
    def setUp(self):
        context = driver.get_or_create_context()
        self.device = context.device
        self.driver = context.driver

    def _template(self, obj):
        self.assertTrue(driver.is_cuda_memory(obj))
        driver.require_cuda_memory(obj)
        self.assertTrue(obj.driver is self.driver)
        self.assertTrue(obj.device is self.device)
        self.assertTrue(isinstance(obj.device_pointer, (int, long)))

    def test_device_memory(self):
        devmem = driver.DeviceMemory(1024)
        self._template(devmem)

    def test_device_view(self):
        devmem = driver.DeviceMemory(1024)
        self._template(driver.DeviceView(devmem, 10))

    def test_host_alloc(self):
        devmem = driver.HostAllocMemory(1024, map=True)
        self._template(devmem)

    def test_pinned_memory(self):
        ary = numpy.arange(10)
        devmem = driver.PinnedMemory(buffer(ary), ary.ctypes.data,
                                     ary.size * ary.dtype.itemsize,
                                     mapped=True)
        self._template(devmem)

if __name__ == '__main__':
    unittest.main()
