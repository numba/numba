from __future__ import print_function, division, absolute_import

import itertools
from itertools import chain, product, starmap
import sys

from numba import unittest_support as unittest
from numba import jit, typeof
from .support import TestCase


def slice_passing(sl):
    return sl.start, sl.stop, sl.step

def slice_constructor(*args):
    sl = slice(*args)
    return sl.start, sl.stop, sl.step

def slice_indices(s, l):
    return s.indices(l)


class TestSlices(TestCase):

    def test_slice_passing(self):
        """
        Check passing a slice object to a Numba function.
        """
        # NOTE this also checks slice attributes
        def check(a, b, c, d, e, f):
            sl = slice(a, b, c)
            got = cfunc(sl)
            self.assertPreciseEqual(got, (d, e, f))

        maxposint = sys.maxsize
        maxnegint = -maxposint - 1
        cfunc = jit(nopython=True)(slice_passing)

        # Positive steps
        start_cases = [(None, 0), (42, 42), (-1, -1)]
        stop_cases = [(None, maxposint), (9, 9), (-11, -11)]
        step_cases = [(None, 1), (12, 12)]
        for (a, d), (b, e), (c, f) in itertools.product(start_cases,
                                                        stop_cases,
                                                        step_cases):
            check(a, b, c, d, e, f)

        # Negative steps
        start_cases = [(None, maxposint), (42, 42), (-1, -1)]
        stop_cases = [(None, maxnegint), (9, 9), (-11, -11)]
        step_cases = [(-1, -1), (-12, -12)]
        for (a, d), (b, e), (c, f) in itertools.product(start_cases,
                                                        stop_cases,
                                                        step_cases):
            check(a, b, c, d, e, f)

        # Some member is neither integer nor None
        with self.assertRaises(TypeError):
            cfunc(slice(1.5, 1, 1))

    def test_slice_constructor(self):
        """
        Test the slice() constructor in nopython mode.
        """
        maxposint = sys.maxsize
        maxnegint = -maxposint - 1
        cfunc = jit(nopython=True)(slice_constructor)
        for args, expected in [((), (0, maxposint, 1)),
                               ((None, None), (0, maxposint, 1)),
                               ((1, None), (1, maxposint, 1)),
                               ((None, 2), (0, 2, 1)),
                               ((1, 2), (1, 2, 1)),
                               ((None, None, 3), (0, maxposint, 3)),
                               ((None, 2, 3), (0, 2, 3)),
                               ((1, None, 3), (1, maxposint, 3)),
                               ((1, 2, 3), (1, 2, 3)),
                               ((None, None, -1), (maxposint, maxnegint, -1)),
                               ((10, None, -1), (10, maxnegint, -1)),
                               ((None, 5, -1), (maxposint, 5, -1)),
                               ((10, 5, -1), (10, 5, -1)),
                               ]:
            got = cfunc(*args)
            self.assertPreciseEqual(got, expected)

    def test_slice_indices(self):
        """Test that a numba slice returns same result for .indices as a python one."""
        slices = starmap(
            slice,
            product(
                chain(range(-5, 5), (None,)),
                chain(range(-5, 5), (None,)),
                chain(range(-5, 5), (None,))
            )
        )
        lens = range(-1, 5)

        cfunc = jit(nopython=True)(slice_indices)

        for s, l in product(slices, lens):
            try:
                expected = slice_indices(s, l)
            except Exception as e:
                self.assertRaises(type(e), cfunc, s, l)
                continue
            self.assertPreciseEqual(expected, cfunc(s, l))


if __name__ == '__main__':
    unittest.main()
