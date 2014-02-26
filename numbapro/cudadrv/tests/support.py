import unittest
from numbapro import testsupport
testsupport.set_base(globals())


class CUDATestCase(unittest.TestCase):
    def tearDown(self):
        from numbapro.cudadrv.devices import reset
        reset()
