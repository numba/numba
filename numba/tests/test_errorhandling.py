"""
Unspecified error handling tests
"""
from __future__ import division

from numba import jit, njit
from numba import unittest_support as unittest
from numba import errors, utils
import numpy as np


class TestErrorHandlingBeforeLowering(unittest.TestCase):

    expected_msg = ("Numba encountered the use of a language feature it does "
                    "not support in this context: %s")

    def test_unsupported_make_function_lambda(self):
        def func(x):
            f = lambda x: x  # requires `make_function`

        for pipeline in jit, njit:
            with self.assertRaises(errors.UnsupportedError) as raises:
                pipeline(func)(1)

            expected = self.expected_msg % "<lambda>"
            self.assertIn(expected, str(raises.exception))

    def test_unsupported_make_function_return_inner_func(self):
        def func(x):
            """ return the closure """
            z = x + 1

            def inner(x):
                return x + z
            return inner

        for pipeline in jit, njit:
            with self.assertRaises(errors.UnsupportedError) as raises:
                pipeline(func)(1)

            expected = self.expected_msg % \
                "<creating a function from a closure>"
            self.assertIn(expected, str(raises.exception))


class TestUnsupportedReporting(unittest.TestCase):

    def test_unsupported_numpy_function(self):
        # np.asarray(list) currently unsupported
        @njit
        def func():
            np.asarray([1,2,3])

        with self.assertRaises(errors.TypingError) as raises:
            func()

        expected = "Use of unsupported NumPy function 'numpy.asarray'"
        self.assertIn(expected, str(raises.exception))

if __name__ == '__main__':
    unittest.main()
