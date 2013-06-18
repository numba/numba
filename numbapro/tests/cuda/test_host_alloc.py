import unittest
import numpy as np
from numbapro.cudadrv import driver
from numbapro import cuda
import support
class TestHostAlloc(support.CudaTestCase):
    def test_host_alloc_driver(self):
        n = 32
        mem = driver.HostAllocMemory(n, mapped=True)

        dtype = np.dtype(np.uint8)
        ary = np.ndarray(shape=n / dtype.itemsize, dtype=dtype, buffer=mem)

        magic = 0xab
        driver.device_memset(mem, magic, n)

        self.assertTrue(np.all(ary == magic))

        ary.fill(n)

        recv = np.empty_like(ary)

        driver.device_to_host(recv, mem, ary.size)

        self.assertTrue(np.all(ary == recv))
        self.assertTrue(np.all(recv == n))

    def test_host_alloc_pinned(self):
        ary = cuda.pinned_array(10, dtype=np.uint32)
        ary.fill(123)
        self.assertTrue(all(ary == 123))
        devary = cuda.to_device(ary)
        driver.device_memset(devary, 0, driver.device_memory_size(devary))
        self.assertTrue(all(ary == 123))
        devary.copy_to_host(ary)
        self.assertTrue(all(ary == 0))

    def test_host_alloc_mapped(self):
        ary = cuda.mapped_array(10, dtype=np.uint32)
        ary.fill(123)
        self.assertTrue(all(ary == 123))
        driver.device_memset(ary, 0, driver.device_memory_size(ary))
        self.assertTrue(all(ary == 0))

    def test_host_alloc_mapped_in_kernel(self):
        ary = cuda.mapped_array(10, dtype=np.uint32)
        ary[:] = np.arange(10, dtype=np.uint32)
        self.assertTrue(all(ary == list(range(10))))
        @cuda.autojit
        def time2(x):
            i = cuda.grid(1)
            x[i] *= 2
        time2[1, ary.size](ary)
        cuda.synchronize()
        self.assertTrue(all(ary == [x * 2 for x in range(10)]))



if __name__ == '__main__':
    unittest.main()
