from __future__ import print_function, division, absolute_import

import numpy as np

from numba.compiler import compile_isolated, DEFAULT_FLAGS
from numba import typeof
from numba.types import float64
from numba import unittest_support as unittest
from .support import MemoryLeakMixin

BOUNDSCHECK_FLAGS = DEFAULT_FLAGS.copy()
BOUNDSCHECK_FLAGS.set('boundscheck', True)

def basic_array_access(a):
    return a[10]

def slice_array_access(a):
    # The first index (slice) is not bounds checked
    return a[10:, 10]


class TestBoundsCheck(MemoryLeakMixin, unittest.TestCase):
    def test_basic_array_boundscheck(self):
        a = np.arange(5)
        # Check the numpy behavior to make sure the test is correct
        with self.assertRaises(IndexError):
            # TODO: When we raise the same error message as numpy, test that
            # they are the same
            basic_array_access(a)

        at = typeof(a)
        c_noboundscheck = compile_isolated(basic_array_access, [at],
                                           flags=DEFAULT_FLAGS)
        noboundscheck = c_noboundscheck.entry_point
        c_boundscheck = compile_isolated(basic_array_access, [at],
                                         flags=BOUNDSCHECK_FLAGS)
        boundscheck = c_boundscheck.entry_point
        # Check that the default flag doesn't raise
        noboundscheck(a)
        with self.assertRaises(IndexError):
            boundscheck(a)

    def test_slice_array_boundscheck(self):
        a = np.ones((5, 5))
        b = np.ones((5, 20))
        with self.assertRaises(IndexError):
            # TODO: When we raise the same error message as numpy, test that
            # they are the same
            slice_array_access(a)
        # Out of bounds on a slice doesn't raise
        slice_array_access(b)

        at = typeof(a)
        rt = float64[:]
        c_noboundscheck = compile_isolated(slice_array_access, [at],
                                           return_type=rt,
                                           flags=DEFAULT_FLAGS)
        noboundscheck = c_noboundscheck.entry_point
        c_boundscheck = compile_isolated(slice_array_access, [at],
                                         return_type=rt,
                                         flags=BOUNDSCHECK_FLAGS)
        boundscheck = c_boundscheck.entry_point
        # Check that the default flag doesn't raise
        noboundscheck(a)
        noboundscheck(b)
        with self.assertRaises(IndexError):
            boundscheck(a)
        # Doesn't raise
        boundscheck(b)

if __name__ == '__main__':
    unittest.main()
