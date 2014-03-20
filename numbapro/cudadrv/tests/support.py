import unittest
from numbapro import testsupport
testsupport.set_base(globals())


class CUDATestCase(unittest.TestCase):
    def tearDown(self):
        from numba.cuda.cudadrv.devices import reset
        reset()
