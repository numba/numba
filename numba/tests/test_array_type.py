from __future__ import print_function, division, absolute_import
import numba.ctypes_support as ctypes

from numba import unittest_support as unittest
from numba import types
from numba.targets.arrayobj import make_array, make_array_ctype
from numba.targets.registry import CPUTarget


class TestArrayType(unittest.TestCase):
    '''
    Tests that the ArrayTemplate returned by make_array and the ctypes array
    struct returned by make_array_ctype both have the same size.
    '''

    def _cpu_array_sizeof(self, ndim):
        ctx = CPUTarget.target_context
        return ctx.calc_array_sizeof(ndim)

    def _test_array(self, ndim):
        c_array = make_array_ctype(ndim)
        self.assertEqual(ctypes.sizeof(c_array),
                         self._cpu_array_sizeof(ndim))

    def test_1d_array(self):
        self._test_array(1)

    def test_2d_array(self):
        self._test_array(2)

    def test_3d_array(self):
        self._test_array(3)


if __name__ == '__main__':
    unittest.main()

