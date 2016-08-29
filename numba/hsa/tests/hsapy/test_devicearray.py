from __future__ import print_function, absolute_import

import numpy as np
from numba import hsa
from numba.hsa.hsadrv.error import HsaDriverError
import numba.unittest_support as unittest


from numba.hsa.hsadrv.driver import dgpu_present


class TestDeviceArray(unittest.TestCase):

    def test_async_copy_to_device(self):
        arr = np.arange(1024)
        devarr = hsa.to_device(arr)

        stream = hsa.stream()
        devarr.copy_to_device(arr + 100, stream=stream)
        # implicit synchronization
        got = devarr.copy_to_host()
        np.testing.assert_equal(arr + 100, got)


if __name__ == '__main__':
    unittest.main()
