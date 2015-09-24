from __future__ import print_function, division, absolute_import

import itertools
import sys

from numba import unittest_support as unittest
from numba import jit, typeof
from .support import TestCase


def slice_passing(sl):
    return sl.start, sl.stop, sl.step


class TestSlices(TestCase):

    def test_slice_passing(self):
        # NOTE this also checks slice attributes
        def check(a, b, c, d, e, f):
            sl = slice(a, b, c)
            got = cfunc(sl)
            self.assertPreciseEqual(got, (d, e, f))

        maxint = sys.maxsize
        cfunc = jit(nopython=True)(slice_passing)
        start_cases = [(None, 0), (42, 42), (-1, -1)]
        stop_cases = [(None, maxint), (9, 9), (-11, -11)]
        step_cases = [(None, 1), (12, 12), (-33, -33)]
        for (a, d), (b, e), (c, f) in itertools.product(start_cases,
                                                        stop_cases,
                                                        step_cases):
            check(a, b, c, d, e, f)

        # Some member is neither integer nor None
        with self.assertRaises(TypeError):
            cfunc(slice(1.5, 1, 1))


if __name__ == '__main__':
    unittest.main()
