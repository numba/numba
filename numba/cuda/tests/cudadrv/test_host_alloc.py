from __future__ import print_function, division, absolute_import
import numpy as np
from numba.cuda.cudadrv import driver
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim


@skip_on_cudasim('CUDA Driver API unsupported in the simulator')
class TestHostAlloc(CUDATestCase):
    def test_host_alloc_driver(self):
        n = 32
        mem = cuda.current_context().memhostalloc(n, mapped=True)

        dtype = np.dtype(np.uint8)
        ary = np.ndarray(shape=n // dtype.itemsize, dtype=dtype,
                         buffer=mem)

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


if __name__ == '__main__':
    unittest.main()
