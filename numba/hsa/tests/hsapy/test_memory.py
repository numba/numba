"""
Test arrays backed by different memory
"""

import logging

import numpy as np

from numba import hsa
import numba.unittest_support as unittest
from numba.hsa.hsadrv.driver import dgpu_present

logger = logging.getLogger()


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
        logger.info('device array like')
        darr = hsa.device_array_like(expect)
        logger.info('pre launch')
        copy_kernel[blkct, blksz](darr, hsa.to_device(expect))
        logger.info('post launch')
        got = darr.copy_to_host()
        np.testing.assert_equal(got, expect)

    def test_coarsegrain_array(self):
        blkct = 4
        blksz = 128
        nelem = blkct * blksz
        expect = np.arange(nelem) + 1
        logger.info('coarsegrain array')
        got = hsa.coarsegrain_array(shape=expect.shape, dtype=expect.dtype)
        got.fill(0)
        logger.info('pre launch')
        copy_kernel[blkct, blksz](got, expect.copy())
        logger.info('post launch')
        np.testing.assert_equal(got, expect)

    def test_finegrain_array(self):
        blkct = 4
        blksz = 128
        nelem = blkct * blksz
        expect = np.arange(nelem) + 1
        logger.info('finegrain array')
        got = hsa.finegrain_array(shape=expect.shape, dtype=expect.dtype)
        got.fill(0)
        logger.info('pre launch')
        copy_kernel[blkct, blksz](got, expect.copy())
        logger.info('post launch')
        np.testing.assert_equal(got, expect)

    def test_device_device_transfer(self):
        nelem = 1000
        expect = np.arange(nelem, dtype=np.int32) + 1
        logger.info('device array like')
        darr = hsa.device_array_like(expect)
        self.assertTrue(np.all(expect != darr.copy_to_host()))
        logger.info('to_device')
        stage = hsa.to_device(expect)
        logger.info('device -> device')
        darr.copy_to_device(stage)
        np.testing.assert_equal(expect, darr.copy_to_host())


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
