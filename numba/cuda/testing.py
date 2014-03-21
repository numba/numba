from __future__ import print_function, absolute_import, division
from numba import unittest_support as unittest


class CUDATestCase(unittest.TestCase):
    def tearDown(self):
        from numba.cuda.cudadrv.devices import reset

        reset()
