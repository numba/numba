import numpy as np
from numba.cuda.cudadrv import driver
from numba import cuda
from numba.cuda.testing import unittest, ContextResettingTestCase
from numba.cuda.testing import skip_on_cudasim


@skip_on_cudasim('CUDA Driver API unsupported in the simulator')
class TestManagedAlloc(ContextResettingTestCase):
    def test_managed_alloc_driver(self):
        # Verify that we can allocate and operate on managed
        # memory through the CUDA driver interface.

        n = 32
        mem = cuda.current_context().memallocmanaged(n)

        dtype = np.dtype(np.uint8)
        ary = np.ndarray(shape=n // dtype.itemsize, dtype=dtype,
                         buffer=mem)

        magic = 0xab
        driver.device_memset(mem, magic, n)

        self.assertTrue(np.all(ary == magic))


    def test_managed_alloc_oversubscription(self):
        # Verify we can correctly operate on a managed array
        # larger than the GPU memory, on both CPU and GPU.

        ctx = cuda.current_context()
        total_mem_size = ctx.get_memory_info().total

        dtype = np.dtype(np.float32)
        n_bytes = 2 * total_mem_size
        n_elems = int(n_bytes / dtype.itemsize)

        ary = cuda.managed_array(n_elems, dtype=dtype)

        ary.fill(123)
        self.assertTrue(all(ary == 123))

        driver.device_memset(ary, 0, n_bytes)
        self.assertTrue(all(ary == 0))


if __name__ == '__main__':
    unittest.main()
