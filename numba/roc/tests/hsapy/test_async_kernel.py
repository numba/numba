"""
Test async kernel copy
"""

import logging

import numpy as np

from numba import roc
import unittest
from numba.roc.hsadrv.driver import dgpu_present

logger = logging.getLogger()


@unittest.skipUnless(dgpu_present, 'test only on dGPU system')
class TestAsyncKernel(unittest.TestCase):

    def test_manual_stream(self):
        logger.info('context info: %s', roc.get_context().agent)

        @roc.jit("int32[:], int32[:]")
        def add1_kernel(dst, src):
            i = roc.get_global_id(0)
            if i < dst.size:
                dst[i] = src[i] + 1

        blksz = 256
        gridsz = 10**5
        nitems = blksz * gridsz
        ntimes = 500

        arr = np.arange(nitems, dtype=np.int32)

        logger.info('make coarse_arr')
        coarse_arr = roc.coarsegrain_array(shape=arr.shape, dtype=arr.dtype)
        coarse_arr[:] = arr

        logger.info('make coarse_res_arr')
        coarse_res_arr = roc.coarsegrain_array(shape=arr.shape, dtype=arr.dtype)
        coarse_res_arr[:] = 0

        logger.info("make stream")
        stream = roc.stream()

        logger.info('make gpu_res_arr')
        gpu_res_arr = roc.device_array_like(coarse_arr)

        logger.info('make gpu_arr')
        gpu_arr = roc.to_device(coarse_arr, stream=stream)

        for i in range(ntimes):
            logger.info('launch kernel: %d', i)
            add1_kernel[gridsz, blksz, stream](gpu_res_arr, gpu_arr)
            gpu_arr.copy_to_device(gpu_res_arr, stream=stream)

        logger.info('get kernel result')
        gpu_res_arr.copy_to_host(coarse_res_arr, stream=stream)

        logger.info("synchronize")
        stream.synchronize()

        logger.info("compare result")
        np.testing.assert_equal(coarse_res_arr, coarse_arr + ntimes)

    def test_ctx_managed_stream(self):
        logger.info('context info: %s', roc.get_context().agent)

        @roc.jit("int32[:], int32[:]")
        def add1_kernel(dst, src):
            i = roc.get_global_id(0)
            if i < dst.size:
                dst[i] = src[i] + 1

        blksz = 256
        gridsz = 10**5
        nitems = blksz * gridsz
        ntimes = 500

        arr = np.arange(nitems, dtype=np.int32)

        logger.info('make coarse_arr')
        coarse_arr = roc.coarsegrain_array(shape=arr.shape, dtype=arr.dtype)
        coarse_arr[:] = arr

        logger.info('make coarse_res_arr')
        coarse_res_arr = roc.coarsegrain_array(shape=arr.shape, dtype=arr.dtype)
        coarse_res_arr[:] = 0

        logger.info("make stream")
        stream = roc.stream()

        with stream.auto_synchronize():
            logger.info('make gpu_res_arr')
            gpu_res_arr = roc.device_array_like(coarse_arr)

            logger.info('make gpu_arr')
            gpu_arr = roc.to_device(coarse_arr, stream=stream)

            for i in range(ntimes):
                logger.info('launch kernel: %d', i)
                add1_kernel[gridsz, blksz, stream](gpu_res_arr, gpu_arr)
                gpu_arr.copy_to_device(gpu_res_arr, stream=stream)

            logger.info('get kernel result')
            gpu_res_arr.copy_to_host(coarse_res_arr, stream=stream)

        logger.info("synchronize on ctx __exit__")

        logger.info("compare result")
        np.testing.assert_equal(coarse_res_arr, coarse_arr + ntimes)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()
