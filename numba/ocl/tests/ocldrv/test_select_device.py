#
# Test does not work on some cards.
#
from __future__ import print_function, absolute_import, division
import threading
import numpy as np
from numba import ocl
import unittest


class TestSelectDevice(unittest.TestCase):
    def test_select_device(self):
        def newthread():
            ocl.select_device(0)
            stream = ocl.stream()

            A = np.arange(100)
            dA = ocl.to_device(A, stream=stream)
            stream.synchronize()
            del dA
            del stream
            ocl.close()

        for i in range(10):
            t = threading.Thread(target=newthread)
            t.start()
            t.join()


if __name__ == '__main__':
    unittest.main()

