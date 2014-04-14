from __future__ import print_function, absolute_import, division
import threading
from numba import cuda
from numba.cuda.cudadrv.driver import driver
from numba.cuda.testing import unittest, CUDATestCase


class TestResetDevice(CUDATestCase):
    def test_reset_device(self):

        def newthread():
            devices = range(driver.get_device_count())
            print('Devices', devices)
            for _ in range(2):
                for d in devices:
                    cuda.select_device(d)
                    print('Selected device', d)
                    cuda.close()
                    print('Closed device', d)

        # Do test on a separate thread so that we don't affect
        # the current context in the main thread.
        t = threading.Thread(target=newthread)
        t.start()
        t.join()


if __name__ == '__main__':
    unittest.main()
