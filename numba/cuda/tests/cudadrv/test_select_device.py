#
# Test does not work on some cards.
#
from __future__ import print_function, absolute_import, division
import threading
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase


def newthread():
    cuda.select_device(0)
    stream = cuda.stream()
    A = np.arange(100)
    dA = cuda.to_device(A, stream=stream)
    stream.synchronize()
    del dA
    del stream
    cuda.close()


class TestSelectDevice(CUDATestCase):
    def test_select_device(self):
        for i in range(10):
            t = threading.Thread(target=newthread)
            t.start()
            t.join()


if __name__ == '__main__':
    unittest.main()

