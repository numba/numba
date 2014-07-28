from __future__ import print_function, division, absolute_import
from numba import unittest_support as unittest
from numba.special import typeof
from numba import vectorize, types, jit
import numpy


def dummy(x):
    return x


class TestDispatcher(unittest.TestCase):

    def test_typeof(self):
        self.assertEqual(typeof(numpy.int8(1)), types.int8)
        self.assertEqual(typeof(numpy.uint16(1)), types.uint16)
        self.assertEqual(typeof(numpy.float64(1)), types.float64)
        self.assertEqual(typeof(numpy.complex128(1)), types.complex128)

    def test_numba_interface(self):
        """
        Check that vectorize can accept a decorated object.
        """
        vectorize('f8(f8)')(jit(dummy))

    def test_no_argument(self):
        @jit
        def foo():
            return 1

        # Just make sure this doesn't crash
        foo()

    def test_inspect_types(self):
        @jit
        def foo(a, b):
            return a + b

        foo(1, 2)
        # Exercise the method
        foo.inspect_types()



if __name__ == '__main__':
    unittest.main()
