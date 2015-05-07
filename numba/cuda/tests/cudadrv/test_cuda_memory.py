import ctypes
import numpy
from numba.cuda.cudadrv import driver, drvapi, devices
from numba.cuda.testing import unittest, CUDATestCase
from numba.utils import IS_PY3
from numba.cuda.testing import skip_on_cudasim


@skip_on_cudasim('CUDA Memory API unsupported in the simulator')
class TestCudaMemory(CUDATestCase):
    def setUp(self):
        self.context = devices.get_context()

    def tearDown(self):
        del self.context

    def _template(self, obj):
        self.assertTrue(driver.is_device_memory(obj))
        driver.require_device_memory(obj)
        self.assertTrue(isinstance(obj.device_ctypes_pointer,
                                   drvapi.cu_device_ptr))

    def test_device_memory(self):
        devmem = self.context.memalloc(1024)
        self._template(devmem)

    def test_device_view(self):
        devmem = self.context.memalloc(1024)
        self._template(devmem.view(10))

    def test_host_alloc(self):
        devmem = self.context.memhostalloc(1024, mapped=True)
        self._template(devmem)

    def test_pinned_memory(self):
        ary = numpy.arange(10)
        arybuf = ary if IS_PY3 else buffer(ary)
        devmem = self.context.mempin(arybuf, ary.ctypes.data,
                                     ary.size * ary.dtype.itemsize,
                                     mapped=True)
        self._template(devmem)


@skip_on_cudasim('CUDA Memory API unsupported in the simulator')
class TestCudaMemoryFunctions(CUDATestCase):
    def setUp(self):
        self.context = devices.get_context()

    def tearDown(self):
        del self.context

    def test_memcpy(self):
        hstary = numpy.arange(100, dtype=numpy.uint32)
        hstary2 = numpy.arange(100, dtype=numpy.uint32)
        sz = hstary.size * hstary.dtype.itemsize
        devary = self.context.memalloc(sz)

        driver.host_to_device(devary, hstary, sz)
        driver.device_to_host(hstary2, devary, sz)

        self.assertTrue(numpy.all(hstary == hstary2))

    def test_memset(self):
        dtype = numpy.dtype('uint32')
        n = 10
        sz = dtype.itemsize * 10
        devary = self.context.memalloc(sz)
        driver.device_memset(devary, 0xab, sz)

        hstary = numpy.empty(n, dtype=dtype)
        driver.device_to_host(hstary, devary, sz)

        hstary2 = numpy.array([0xabababab] * n, dtype=numpy.dtype('uint32'))
        self.assertTrue(numpy.all(hstary == hstary2))

    def test_d2d(self):
        hst = numpy.arange(100, dtype=numpy.uint32)
        hst2 = numpy.empty_like(hst)
        sz = hst.size * hst.dtype.itemsize
        dev1 = self.context.memalloc(sz)
        dev2 = self.context.memalloc(sz)
        driver.host_to_device(dev1, hst, sz)
        driver.device_to_device(dev2, dev1, sz)
        driver.device_to_host(hst2, dev2, sz)
        self.assertTrue(numpy.all(hst == hst2))


@skip_on_cudasim('CUDA Memory API unsupported in the simulator')
class TestMVExtent(CUDATestCase):
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
    unittest.main()
