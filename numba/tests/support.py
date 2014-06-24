"""
Assorted utilities for use in tests.
"""

import contextlib

from numba.lowering import LoweringError
from numba.typeinfer import TypingError
import numba.unittest_support as unittest


class TestCase(unittest.TestCase):

    @contextlib.contextmanager
    def assertTypingError(self):
        """
        A context manager that asserts the enclosed code block fails
        compiling in nopython mode.
        """
        with self.assertRaises(
            (LoweringError, TypingError, TypeError, NotImplementedError)) as cm:
            yield cm

