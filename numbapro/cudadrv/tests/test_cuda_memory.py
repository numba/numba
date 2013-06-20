import unittest
import ctypes
import numpy
from numbapro.cudadrv import driver
from .support import addtest, main

@addtest
class TestCudaMemory(unittest.TestCase):
    def setUp(self):
        context = driver.get_or_create_context()
        self.device = context.device
        self.driver = context.driver

    def _template(self, obj):
        self.assertTrue(driver.is_device_memory(obj))
        driver.require_device_memory(obj)
        self.assertTrue(obj.driver is self.driver)
        self.assertTrue(obj.device is self.device)
        self.assertTrue(isinstance(obj.device_ctypes_pointer,
                                   driver.cu_device_ptr))

    def test_device_memory(self):
        devmem = driver.DeviceMemory(1024)
        self._template(devmem)

    def test_device_view(self):
        devmem = driver.DeviceMemory(1024)
        self._template(driver.DeviceView(devmem, 10))

    def test_host_alloc(self):
        devmem = driver.HostAllocMemory(1024, mapped=True)
        self._template(devmem)

    def test_pinned_memory(self):
        ary = numpy.arange(10)
        devmem = driver.PinnedMemory(buffer(ary), ary.ctypes.data,
                                     ary.size * ary.dtype.itemsize,
                                     mapped=True)
        self._template(devmem)

class TestCudaMemoryFunctions(unittest.TestCase):
    def setUp(self):
        driver.get_or_create_context()

    def test_memcpy(self):
        hstary = numpy.arange(100, dtype=numpy.uint32)
        hstary2 = numpy.arange(100, dtype=numpy.uint32)
        sz = hstary.size * hstary.dtype.itemsize
        devary = driver.DeviceMemory(sz)

        driver.host_to_device(devary, hstary, sz)
        driver.device_to_host(hstary2, devary, sz)

        self.assertTrue(numpy.all(hstary == hstary2))

    def test_memset(self):
        dtype = numpy.dtype('uint32')
        n = 10
        sz = dtype.itemsize * 10
        devary = driver.DeviceMemory(sz)
        driver.device_memset(devary, 0xab, sz)

        hstary = numpy.empty(n, dtype=dtype)
        driver.device_to_host(hstary, devary, sz)

        hstary2 = numpy.array([0xabababab] * n, dtype=numpy.dtype('uint32'))
        self.assertTrue(numpy.all(hstary == hstary2))

    def test_d2d(self):
        hst = numpy.arange(100, dtype=numpy.uint32)
        hst2 = numpy.empty_like(hst)
        sz = hst.size * hst.dtype.itemsize
        dev1 = driver.DeviceMemory(sz)
        dev2 = driver.DeviceMemory(sz)
        driver.host_to_device(dev1, hst, sz)
        driver.device_to_device(dev2, dev1, sz)
        driver.device_to_host(hst2, dev2, sz)
        self.assertTrue(numpy.all(hst == hst2))

@addtest
class TestMVExtent(unittest.TestCase):
    def test_c_contiguous_array(self):
        ary = numpy.arange(100)
        arysz = ary.dtype.itemsize * ary.size
        s, e = driver.host_memory_extents(ary)
        self.assertTrue(ary.ctypes.data == s)
        self.assertTrue(arysz == driver.host_memory_size(ary))

    def test_f_contiguous_array(self):
        ary = numpy.asfortranarray(numpy.arange(100).reshape(2, 50))
        arysz = ary.dtype.itemsize * numpy.prod(ary.shape)
        s, e = driver.host_memory_extents(ary)
        self.assertTrue(ary.ctypes.data == s)
        self.assertTrue(arysz == driver.host_memory_size(ary))

    def test_single_element_array(self):
        ary = numpy.asarray(numpy.uint32(1234))
        arysz = ary.dtype.itemsize
        s, e = driver.host_memory_extents(ary)
        self.assertTrue(ary.ctypes.data == s)
        self.assertTrue(arysz == driver.host_memory_size(ary))

    def test_ctypes_struct(self):
        class mystruct(ctypes.Structure):
            _fields_ = [('x', ctypes.c_int), ('y', ctypes.c_int)]
        data = mystruct(x=123, y=432)
        sz = driver.host_memory_size(data)
        self.assertTrue(ctypes.sizeof(data) == sz)

    def test_ctypes_double(self):
        data = ctypes.c_double(1.234)
        sz = driver.host_memory_size(data)
        self.assertTrue(ctypes.sizeof(data) == sz)


if __name__ == '__main__':
    main()
