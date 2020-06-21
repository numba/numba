import numpy as np
import platform
from numba.cuda.cudadrv import driver
from numba import cuda
from numba.cuda.testing import unittest, ContextResettingTestCase
from numba.cuda.testing import skip_on_cudasim


@skip_on_cudasim('CUDA Driver API unsupported in the simulator')
class TestManagedAlloc(ContextResettingTestCase):
    def test_managed_alloc_driver(self):
        # Verify that we can allocate and operate on managed
        # memory through the CUDA driver interface.

        ctx = cuda.current_context()

        # CUDA Unified Memory comes in two flavors. For GPUs in the
        # NVIDIA Kepler and Maxwell generations, managed memory
        # allocations work as opaque, contiguous segments that can
        # either be on the device or the host. For GPUs in the Pascal
        # or later generations, managed memory operates on a per-page
        # basis, so we can have arrays larger than GPU memory, where
        # only part of them is resident on the device at one time. To
        # ensure that this test works correctly on all supported GPUs,
        # we'll select the size of our memory such that we only
        # oversubscribe the GPU memory if we're on a Pascal or newer GPU
        # (compute capability at least 6.0).

        compute_capability_major = ctx.device.compute_capability[0]

        # This test is unsupported on GPUs prior to the Kepler generation.

        self.assertGreaterEqual(compute_capability_major, 3)

        total_mem_size = ctx.get_memory_info().total

        if compute_capability_major >= 6 and platform.system() == 'Linux':
            n_bytes = int(2 * total_mem_size)
        else:
            n_bytes = int(0.5 * total_mem_size)

        dtype = np.dtype(np.uint8)
        n_elems = n_bytes // dtype.itemsize

        mem = ctx.memallocmanaged(n_bytes)

        ary = np.ndarray(shape=n_elems, dtype=dtype, buffer=mem)

        magic = 0xab
        driver.device_memset(mem, magic, n_bytes)
        ctx.synchronize()

        # Note that this assertion operates on the CPU, so this
        # test effectively drives both the CPU and the GPU on
        # managed memory.

        self.assertTrue(np.all(ary == magic))

    def test_managed_array(self):
        # Check the managed_array interface on both host and device.

        ary = cuda.managed_array(100, dtype=np.double)
        ary.fill(123.456)
        self.assertTrue(all(ary == 123.456))

        @cuda.jit('void(double[:])')
        def kernel(x):
            i = cuda.grid(1)
            if i < x.shape[0]:
                x[i] = 1.0

        kernel[10, 10](ary)
        cuda.current_context().synchronize()

        self.assertTrue(all(ary == 1.0))


if __name__ == '__main__':
    unittest.main()
