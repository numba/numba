"""
Test arrays backed by different memory
"""

import numpy as np

from numba import hsa
import numba.unittest_support as unittest
from numba.hsa.hsadrv.driver import dgpu_present


@hsa.jit
def copy_kernel(dst, src):
    i = hsa.get_global_id(0)
    if i < dst.size:
        dst[i] = src[i]


@unittest.skipUnless(dgpu_present(), 'test only on dGPU system')
class TestMemory(unittest.TestCase):
    def test_auto_device(self):
        blkct = 4
        blksz = 128
        nelem = blkct * blksz
        expect = np.arange(nelem) + 1
        got = np.zeros_like(expect)
        copy_kernel[blkct, blksz](got, expect.copy())
        np.testing.assert_equal(got, expect)

    def test_device_array(self):
        blkct = 4
        blksz = 128
        nelem = blkct * blksz
        expect = np.arange(nelem) + 1
        darr = hsa.device_array_like(expect)
        copy_kernel[blkct, blksz](darr, hsa.to_device(expect))
        got = darr.copy_to_host()
        np.testing.assert_equal(got, expect)

    def test_coarsegrain_array(self):
        blkct = 4
        blksz = 128
        nelem = blkct * blksz
        expect = np.arange(nelem) + 1
        got = hsa.coarsegrain_array(shape=expect.shape, dtype=expect.dtype)
        got.fill(0)
        copy_kernel[blkct, blksz](got, expect.copy())
        np.testing.assert_equal(got, expect)


if __name__ == '__main__':
    unittest.main()
