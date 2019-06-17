from __future__ import print_function, absolute_import

import numpy as np

from numba import roc
import numba.unittest_support as unittest
from numba.roc.hsadrv.driver import dgpu_present


@unittest.skipUnless(dgpu_present, 'test only on dGPU system')
class TestAsync(unittest.TestCase):

    def test_coarsegrain_array(self):
        arr = roc.coarsegrain_array(shape=1024, dtype=np.float32)
        self.assertEqual(arr.size, 1024)
        arr[:] = expect = np.arange(arr.size)
        np.testing.assert_allclose(arr, expect)

    def test_async_copy_to_device(self):
        arr = np.arange(1024)

        devarr = roc.to_device(arr)

        # allocate pinned array equivalent
        hostarr = roc.coarsegrain_array(shape=arr.shape, dtype=arr.dtype)
        hostarr[:] = arr + 100

        stream = roc.stream()
        ct = len(stream._signals)
        devarr.copy_to_device(hostarr, stream=stream)
        self.assertEqual(ct + 1, len(stream._signals),
                         "no new async signal")
        # implicit synchronization
        got = devarr.copy_to_host()
        self.assertEqual(0, len(stream._signals),
                         "missing implicit synchronization")
        np.testing.assert_equal(hostarr, got)

    def test_async_copy_to_device_and_back(self):
        arr = np.arange(1024)
        hostarr = roc.coarsegrain_array(shape=arr.shape, dtype=arr.dtype)
        gotarr = roc.coarsegrain_array(shape=arr.shape, dtype=arr.dtype)
        stream = roc.stream()
        ct = len(stream._signals)
        devarr = roc.to_device(hostarr, stream=stream)
        self.assertEqual(ct + 1, len(stream._signals))
        devarr.copy_to_host(gotarr, stream=stream)
        self.assertEqual(ct + 2, len(stream._signals))
        stream.synchronize()
        self.assertEqual(0, len(stream._signals))
        np.testing.assert_equal(hostarr, gotarr)


if __name__ == '__main__':
    unittest.main()
