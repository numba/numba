from __future__ import print_function, division, absolute_import
from numba import unittest_support as unittest
from numba import types
from numba import njit

class TestOmittedArgs(unittest.TestCase):

    def test_error_lowering_constant(self):
        """
        Issue 1868
        """

        def pyfunc(x=None):
            if x is None:
                x = 1
            return x

        cfunc = njit(pyfunc)
        self.assertEqual(pyfunc(), cfunc())
        self.assertEqual(pyfunc(3), cfunc(3))


if __name__ == '__main__':
    unittest.main()
