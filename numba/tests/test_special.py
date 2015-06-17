"""
Basic tests for the numba.special module.
"""

from __future__ import print_function

import numpy as np

import numba.unittest_support as unittest
from numba import special, types
from .support import TestCase
from .test_numpy_support import ValueTypingTestBase


class TestArrayScalars(ValueTypingTestBase, TestCase):

    def test_number_values(self):
        """
        Test special.typeof() with scalar number values.
        """
        self.check_number_values(special.typeof)

    def test_datetime_values(self):
        """
        Test special.typeof() with np.timedelta64 values.
        """
        self.check_datetime_values(special.typeof)

    def test_timedelta_values(self):
        """
        Test special.typeof() with np.timedelta64 values.
        """
        self.check_timedelta_values(special.typeof)

    def test_array_values(self):
        """
        Test special.typeof() with ndarray values.
        """
        def check(arr, ndim, layout, mutable):
            ty = special.typeof(arr)
            self.assertIsInstance(ty, types.Array)
            self.assertEqual(ty.ndim, ndim)
            self.assertEqual(ty.layout, layout)
            self.assertEqual(ty.mutable, mutable)

        a1 = np.arange(10)
        check(a1, 1, 'C', True)
        a2 = np.arange(10).reshape(2, 5)
        check(a2, 2, 'C', True)
        check(a2.T, 2, 'F', True)
        a3 = (np.arange(60))[::2].reshape((2, 5, 3))
        check(a3, 3, 'A', True)
        a4 = np.arange(1).reshape(())
        check(a4, 0, 'C', True)
        a4.flags.writeable = False
        check(a4, 0, 'C', False)


if __name__ == '__main__':
    unittest.main()
