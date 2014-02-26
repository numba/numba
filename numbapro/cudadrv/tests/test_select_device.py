#
# Test does not work on some cards.
#
from __future__ import print_function, absolute_import, division
import threading
import numpy as np
from . import support
from numbapro import cuda

def newthread():
    cuda.select_device(0)
    stream = cuda.stream()
    A = np.arange(100)
    dA = cuda.to_device(A, stream=stream)
    stream.synchronize()
    del dA
    del stream
    cuda.close()


@support.addtest
class TestSelectDevice(support.CUDATestCase):
    def test_select_device(self):
        for i in range(10):
            t = threading.Thread(target=newthread)
            t.start()
            t.join()


if __name__ == '__main__':
    support.main()

