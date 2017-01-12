import ctypes

import numpy as np

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
        ary = np.arange(10)
        arybuf = ary if IS_PY3 else buffer(ary)
        devmem = self.context.mempin(arybuf, ary.ctypes.data,
                                     ary.size * ary.dtype.itemsize,
                                     mapped=True)
        self._template(devmem)

    def test_derived_pointer(self):
        # Use MemoryPointer.view to create derived pointer
        def check(m, offset):
            # create view
            v1 = m.view(offset)
            self.assertEqual(v1.owner.handle.value, m.handle.value)
            self.assertEqual(m.refct, 2)
            self.assertEqual(v1.handle.value - offset, v1.owner.handle.value)
            # create a view
            v2 = v1.view(offset)
            self.assertEqual(v2.owner.handle.value, m.handle.value)
            self.assertEqual(v2.owner.handle.value, m.handle.value)
            self.assertEqual(v2.handle.value - offset * 2,
                             v2.owner.handle.value)
            self.assertEqual(m.refct, 3)
            del v2
            self.assertEqual(m.refct, 2)
            del v1
            self.assertEqual(m.refct, 1)

        m = self.context.memalloc(1024)
        check(m=m, offset=0)
        check(m=m, offset=1)


@skip_on_cudasim('CUDA Memory API unsupported in the simulator')
class TestCudaMemoryFunctions(CUDATestCase):
    def setUp(self):
        self.context = devices.get_context()

    def tearDown(self):
        del self.context

    def test_memcpy(self):
        hstary = np.arange(100, dtype=np.uint32)
        hstary2 = np.arange(100, dtype=np.uint32)
        sz = hstary.size * hstary.dtype.itemsize
        devary = self.context.memalloc(sz)

        driver.host_to_device(devary, hstary, sz)
        driver.device_to_host(hstary2, devary, sz)

        self.assertTrue(np.all(hstary == hstary2))

    def test_memset(self):
        dtype = np.dtype('uint32')
        n = 10
        sz = dtype.itemsize * 10
        devary = self.context.memalloc(sz)
        driver.device_memset(devary, 0xab, sz)

        hstary = np.empty(n, dtype=dtype)
        driver.device_to_host(hstary, devary, sz)

        hstary2 = np.array([0xabababab] * n, dtype=np.dtype('uint32'))
        self.assertTrue(np.all(hstary == hstary2))

    def test_d2d(self):
        hst = np.arange(100, dtype=np.uint32)
        hst2 = np.empty_like(hst)
        sz = hst.size * hst.dtype.itemsize
        dev1 = self.context.memalloc(sz)
        dev2 = self.context.memalloc(sz)
        driver.host_to_device(dev1, hst, sz)
        driver.device_to_device(dev2, dev1, sz)
        driver.device_to_host(hst2, dev2, sz)
        self.assertTrue(np.all(hst == hst2))


@skip_on_cudasim('CUDA Memory API unsupported in the simulator')
class TestMVExtent(CUDATestCase):
    def test_c_contiguous_array(self):
        ary = np.arange(100)
        arysz = ary.dtype.itemsize * ary.size
        s, e = driver.host_memory_extents(ary)
        self.assertTrue(ary.ctypes.data == s)
        self.assertTrue(arysz == driver.host_memory_size(ary))

    def test_f_contiguous_array(self):
        ary = np.asfortranarray(np.arange(100).reshape(2, 50))
        arysz = ary.dtype.itemsize * np.prod(ary.shape)
        s, e = driver.host_memory_extents(ary)
        self.assertTrue(ary.ctypes.data == s)
        self.assertTrue(arysz == driver.host_memory_size(ary))

    def test_single_element_array(self):
        ary = np.asarray(np.uint32(1234))
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
