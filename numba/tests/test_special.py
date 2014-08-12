"""
Basic tests for the numba.special module.
"""

from __future__ import print_function

import numpy as np

import numba.unittest_support as unittest
from numba import special
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



if __name__ == '__main__':
    unittest.main()
