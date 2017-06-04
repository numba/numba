from __future__ import print_function, absolute_import, division

from numba.test_utils import InOtherThread
from numba import ocl
import numpy as np
import numba.unittest_support as unittest

class TestSelectDevice(unittest.TestCase):
    @unittest.skip('not yet implemented')
    def test_select_device(self):
        def newthread():
            ocl.select_device(0)
            stream = ocl.stream()

            A = np.arange(100)
            dA = ocl.to_device(A, stream=stream)
            stream.synchronize()
            del dA
            del stream
            assert False
            ocl.close()

        for i in range(10):
            InOtherThread(newthread).return_value


if __name__ == '__main__':
    unittest.main()