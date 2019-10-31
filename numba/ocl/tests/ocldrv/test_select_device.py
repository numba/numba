#
# Test does not work on some cards.
#
from __future__ import print_function, absolute_import, division
import threading
try:
    from Queue import Queue  # Python 2
except:
    from queue import Queue  # Python 3

import numpy as np
from numba import ocl
from numba.ocl.testing import unittest, OCLTestCase


def newthread(exception_queue):
    try:
        ocl.select_platform(0)
        ocl.select_device(0)
        stream = ocl.stream()
        A = np.arange(100)
        dA = ocl.to_device(A, stream=stream)
        stream.finish()
        del dA
        del stream
        ocl.close()
    except Exception as e:
        exception_queue.put(e)


class TestSelectDevice(OCLTestCase):
    def test_select_device(self):
        exception_queue = Queue()
        for i in range(10):
            t = threading.Thread(target=newthread, args=(exception_queue,))
            t.start()
            t.join()

        exceptions = []
        while not exception_queue.empty():
            exceptions.append(exception_queue.get())
        self.assertEqual(exceptions, [])


if __name__ == '__main__':
    unittest.main()

