from numbapro import cuda
from numbapro.cudadrv import driver
import unittest, threading

from . import support

@support.addtest
class TestResetDevice(unittest.TestCase):
    def test_reset_device(self):

        def newthread():
            drv = driver.Driver()
            devices = range(drv.get_device_count())
            print 'Devices', devices
            for _ in range(2):
                for d in devices:
                    cuda.select_device(d)
                    print 'Selected device', d
                    cuda.close()
                    print 'Closed device', d

        # Do test on a separate thread so that we don't affect
        # the current context in the main thread.
        t = threading.Thread(target=newthread)
        t.start()
        t.join()

if __name__ == '__main__':
    support.main()
