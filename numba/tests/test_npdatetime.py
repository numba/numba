"""
Test numpy.timedelta64 support.
"""

from __future__ import print_function

import itertools

import numpy as np

import numba.unittest_support as unittest
from numba.typeinfer import TypingError
from numba import config, jit, npdatetime, types
from .support import TestCase


date_units = ('Y', 'M')
time_units = ('W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs', 'as')
# All except generic ("")
all_units = date_units + time_units


def add_usecase(x, y):
    return x + y

def sub_usecase(x, y):
    return x - y

def mul_usecase(x, y):
    return x * y

def div_usecase(x, y):
    return x / y

def floordiv_usecase(x, y):
    return x // y


class TestModuleHelpers(TestCase):
    """
    Test the various helpers in numba.npdatetime.
    """

    def test_can_cast_timedelta(self):
        f = npdatetime.can_cast_timedelta_units
        for a, b in itertools.product(date_units, time_units):
            self.assertFalse(f(a, b), (a, b))
            self.assertFalse(f(b, a), (a, b))
        for unit in all_units:
            self.assertFalse(f(unit, ''))
            self.assertTrue(f('', unit))
        for unit in all_units + ('',):
            self.assertTrue(f(unit, unit))

        def check_units_group(group):
            for i, a in enumerate(group):
                for b in group[:i]:
                    # large into smaller is ok
                    self.assertTrue(f(b, a))
                    # small into larger is not
                    self.assertFalse(f(a, b))

        check_units_group(date_units)
        check_units_group(time_units)

    def test_timedelta_conversion(self):
        f = npdatetime.get_timedelta_conversion_factor
        for unit in all_units + ('',):
            self.assertEqual(f(unit, unit), 1)
        for unit in all_units:
            self.assertEqual(f('', unit), 1)
        for a, b in itertools.product(time_units, date_units):
            self.assertIs(f(a, b), None)
            self.assertIs(f(b, a), None)

        def check_units_group(group):
            for i, a in enumerate(group):
                for b in group[:i]:
                    self.assertGreater(f(b, a), 1, (b, a))
                    self.assertIs(f(a, b), None)

        check_units_group(date_units)
        check_units_group(time_units)

        # Check some hand-picked values
        self.assertEqual(f('Y', 'M'), 12)
        self.assertEqual(f('W', 'h'), 24 * 7)
        self.assertEqual(f('W', 'm'), 24 * 7 * 60)
        self.assertEqual(f('W', 'us'), 24 * 7 * 3600 * 1000 * 1000)


TD = np.timedelta64
DT = np.datetime64


class TestScalarOperators(TestCase):

    jitargs = dict()

    def jit(self, pyfunc):
        return jit(**self.jitargs)(pyfunc)

    def test_add(self):
        f = self.jit(add_usecase)
        def check(a, b, expected):
            self.assertPreciseEqual(f(a, b), expected)
            self.assertPreciseEqual(f(b, a), expected)

        check(TD(1), TD(2), TD(3))
        check(TD(1, 's'), TD(2, 's'), TD(3, 's'))
        # Implicit unit promotion
        check(TD(1), TD(2, 's'), TD(3, 's'))
        check(TD(1), TD(2, 'ms'), TD(3, 'ms'))
        check(TD(1, 's'), TD(2, 'us'), TD(1000002, 'us'))
        check(TD(1, 'W'), TD(2, 'D'), TD(9, 'D'))
        # NaTs
        check(TD('NaT'), TD(1), TD('NaT'))
        check(TD('NaT', 's'), TD(1, 'D'), TD('NaT', 's'))
        check(TD('NaT', 's'), TD(1, 'ms'), TD('NaT', 'ms'))
        # Cannot add days and weeks
        with self.assertRaises((TypeError, TypingError)):
            f(TD(1, 'M'), TD(1, 'D'))

    def test_sub(self):
        f = self.jit(sub_usecase)
        def check(a, b, expected):
            self.assertPreciseEqual(f(a, b), expected)
            self.assertPreciseEqual(f(b, a), -expected)

        check(TD(3), TD(2), TD(1))
        check(TD(3, 's'), TD(2, 's'), TD(1, 's'))
        # Implicit unit promotion
        check(TD(3), TD(2, 's'), TD(1, 's'))
        check(TD(3), TD(2, 'ms'), TD(1, 'ms'))
        check(TD(3, 's'), TD(2, 'us'), TD(2999998, 'us'))
        check(TD(1, 'W'), TD(2, 'D'), TD(5, 'D'))
        # NaTs
        check(TD('NaT'), TD(1), TD('NaT'))
        check(TD('NaT', 's'), TD(1, 'D'), TD('NaT', 's'))
        check(TD('NaT', 's'), TD(1, 'ms'), TD('NaT', 'ms'))
        # Cannot sub days to weeks
        with self.assertRaises((TypeError, TypingError)):
            f(TD(1, 'M'), TD(1, 'D'))

    def test_mul(self):
        f = self.jit(mul_usecase)
        def check(a, b, expected):
            self.assertPreciseEqual(f(a, b), expected)
            self.assertPreciseEqual(f(b, a), expected)

        # int * timedelta64
        check(TD(3), 2, TD(6))
        check(TD(3, 'ps'), 2, TD(6, 'ps'))
        check(TD('NaT', 'ps'), 2, TD('NaT', 'ps'))
        # float * timedelta64
        check(TD(7), 1.5, TD(10))
        check(TD(-7), 1.5, TD(-10))
        check(TD(7, 'ps'), -1.5, TD(-10, 'ps'))
        check(TD(-7), -1.5, TD(10))
        check(TD('NaT', 'ps'), -1.5, TD('NaT', 'ps'))
        check(TD(7, 'ps'), float('nan'), TD('NaT', 'ps'))
        # wraparound on overflow
        check(TD(2**62, 'ps'), 16, TD(0, 'ps'))

    def test_div(self):
        div = self.jit(div_usecase)
        floordiv = self.jit(floordiv_usecase)
        def check(a, b, expected):
            self.assertPreciseEqual(div(a, b), expected)
            self.assertPreciseEqual(floordiv(a, b), expected)

        # timedelta64 / int
        check(TD(3), 2, TD(1))
        check(TD(-3, 'ps'), 2, TD(-1, 'ps'))
        check(TD('NaT', 'ps'), 2, TD('NaT', 'ps'))
        check(TD(3, 'ps'), 0, TD('NaT', 'ps'))
        check(TD('NaT', 'ps'), 0, TD('NaT', 'ps'))
        # timedelta64 / float
        check(TD(7), 0.5, TD(14))
        check(TD(-7, 'ps'), 1.5, TD(-4, 'ps'))
        check(TD('NaT', 'ps'), 2.5, TD('NaT', 'ps'))
        check(TD(3, 'ps'), 0.0, TD('NaT', 'ps'))
        check(TD('NaT', 'ps'), 0.0, TD('NaT', 'ps'))
        check(TD(3, 'ps'), float('nan'), TD('NaT', 'ps'))
        check(TD('NaT', 'ps'), float('nan'), TD('NaT', 'ps'))

    def test_homogenous_div(self):
        div = self.jit(div_usecase)
        def check(a, b, expected):
            self.assertPreciseEqual(div(a, b), expected)

        # timedelta64 / timedelta64
        check(TD(7), TD(3), 7. / 3.)
        check(TD(7), TD(3, 'ms'), 7. / 3.)
        check(TD(7, 'us'), TD(3, 'ms'), 7. / 3000.)
        check(TD(7, 'ms'), TD(3, 'us'), 7000. / 3.)
        check(TD(7), TD(0), float('+inf'))
        check(TD(-7), TD(0), float('-inf'))
        # NaTs
        check(TD('nat'), TD(3), float('nan'))
        check(TD(3), TD('nat'), float('nan'))
        check(TD('nat'), TD(0), float('nan'))


class TestScalarOperatorsNoPython(TestScalarOperators):

    jitargs = dict(nopython=True)


if __name__ == '__main__':
    unittest.main()
