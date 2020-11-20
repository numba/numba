import multiprocessing
import os
from numba.core import config
from numba.cuda.cudadrv.runtime import runtime
from numba.cuda.testing import unittest, SerialMixin


def set_visible_devices_and_check(q):
    try:
        from numba import cuda
        import os

        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        q.put(len(cuda.gpus.lst))
    except: # noqa: E722
        # Sentinel value for error executing test code
        q.put(-1)


class TestRuntime(unittest.TestCase):
    def test_get_version(self):
        if config.ENABLE_CUDASIM:
            supported_versions = (-1, -1),
        else:
            supported_versions = ((9, 0), (9, 1), (9, 2), (10, 0),
                                  (10, 1), (10, 2), (11, 0))
        self.assertIn(runtime.get_version(), supported_versions)


class TestVisibleDevices(unittest.TestCase, SerialMixin):
    def test_visible_devices_set_after_import(self):
        # See Issue #6149. This test checks that we can set
        # CUDA_VISIBLE_DEVICES after importing Numba and have the value
        # reflected in the available list of GPUs. Prior to the fix for this
        # issue, Numba made a call to runtime.get_version() on import that
        # initialized the driver and froze the list of available devices before
        # CUDA_VISIBLE_DEVICES could be set by the user.

        # Avoid importing cuda at the top level so that
        # set_visible_devices_and_check gets to import it first in its process
        from numba import cuda

        if len(cuda.gpus.lst) in (0, 1):
            self.skipTest('This test requires multiple GPUs')

        if os.environ.get('CUDA_VISIBLE_DEVICES'):
            msg = 'Cannot test when CUDA_VISIBLE_DEVICES already set'
            self.skipTest(msg)

        ctx = multiprocessing.get_context('spawn')
        q = ctx.Queue()
        p = ctx.Process(target=set_visible_devices_and_check, args=(q,))
        p.start()
        try:
            visible_gpu_count = q.get()
        finally:
            p.join()

        # Make an obvious distinction between an error running the test code
        # and an incorrect number of GPUs in the list
        msg = 'Error running set_visible_devices_and_check'
        self.assertNotEqual(visible_gpu_count, -1, msg=msg)

        # The actual check that we see only one GPU
        self.assertEqual(visible_gpu_count, 1)


if __name__ == '__main__':
    unittest.main()
