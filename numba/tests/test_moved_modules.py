"""Tests for moved modules and their redirection from old path
"""
from numba.tests.support import TestCase


class TestMovedModule(TestCase):
    """Testing moved modules in Q1 2020 but were decided to kept as public API
    """
    def tests_numba_types(self):
        import numba.types
        import numba.core.types as types
        # The old module is the new module
        self.assertIs(numba.types, types)
        # Attribute access are there
        self.assertIs(numba.types.intp, types.intp)
        self.assertIs(numba.types.float64, types.float64)
        self.assertIs(numba.types.Array, types.Array)
        # Submodule access through old import path is possible
        import numba.types.misc
        self.assertIs(types.misc, numba.types.misc)
        self.assertIs(types.misc.Optional, numba.types.misc.Optional)
