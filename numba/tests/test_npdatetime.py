"""
Test numpy.timedelta64 support.
"""

from __future__ import print_function

import itertools

import numpy as np

import numba.unittest_support as unittest
from numba import config, npdatetime, types
from .support import TestCase


date_units = ('Y', 'M')
time_units = ('W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs', 'as')
# All except generic ("")
all_units = date_units + time_units


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


if __name__ == '__main__':
    unittest.main()
