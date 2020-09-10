import numpy as np
import platform
from ctypes import byref, c_size_t
from numba.cuda.cudadrv.driver import device_memset, driver
from numba import cuda
from numba.cuda.testing import unittest, ContextResettingTestCase
from numba.cuda.testing import skip_on_cudasim


@skip_on_cudasim('CUDA Driver API unsupported in the simulator')
class TestManagedAlloc(ContextResettingTestCase):

    def get_total_gpu_memory(self):
        # We use a driver function to directly get the total GPU memory because
        # an EMM plugin may report something different (or not implement
        # get_memory_info at all).
        free = c_size_t()
        total = c_size_t()
        driver.cuMemGetInfo(byref(free), byref(total))
        return total.value

    def skip_if_cc_major_lt(self, min_required, reason):
        """
        Skip the current test if the compute capability of the device is
        less than `min_required`.
        """
        ctx = cuda.current_context()
        cc_major = ctx.device.compute_capability[0]
        if cc_major < min_required:
            self.skipTest(reason)

    # CUDA Unified Memory comes in two flavors. For GPUs in the Kepler and
    # Maxwell generations, managed memory allocations work as opaque,
    # contiguous segments that can either be on the device or the host. For
    # GPUs in the Pascal or later generations, managed memory operates on a
    # per-page basis, so we can have arrays larger than GPU memory, where only
    # part of them is resident on the device at one time. To ensure that this
    # test works correctly on all supported GPUs, we'll select the size of our
    # memory such that we only oversubscribe the GPU memory if we're on a
    # Pascal or newer GPU (compute capability at least 6.0).

    def test_managed_alloc_driver_undersubscribe(self):
        msg = "Managed memory unsupported prior to CC 3.0"
        self.skip_if_cc_major_lt(3, msg)
        self._test_managed_alloc_driver(0.5)

    def test_managed_alloc_driver_oversubscribe(self):
        msg = "Oversubscription of managed memory unsupported prior to CC 6.0"
        self.skip_if_cc_major_lt(6, msg)
        if platform.system() != "Linux":
            msg = "Oversubscription of managed memory only supported on Linux"
            self.skipTest(msg)
        self._test_managed_alloc_driver(2.0)

    def _test_managed_alloc_driver(self, memory_factor):
        # Verify that we can allocate and operate on managed
        # memory through the CUDA driver interface.

        total_mem_size = self.get_total_gpu_memory()
        n_bytes = int(memory_factor * total_mem_size)

        ctx = cuda.current_context()
        mem = ctx.memallocmanaged(n_bytes)

        dtype = np.dtype(np.uint8)
        n_elems = n_bytes // dtype.itemsize
        ary = np.ndarray(shape=n_elems, dtype=dtype, buffer=mem)

        magic = 0xab
        device_memset(mem, magic, n_bytes)
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
