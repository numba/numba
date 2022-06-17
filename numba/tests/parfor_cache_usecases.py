"""
This file will be copied to a temporary directory in order to
exercise caching compiled C callbacks.

See test_cfunc.py.
"""

import sys

from numba import cfunc, njit
import numpy as np
from numba.tests.support import TestCase, captured_stderr


@njit(parallel=True, cache=True)
def f(a, b):
    return np.sum(a + b)


class _TestModule(TestCase):
    """
    Tests for functionality of this module's cfuncs.
    Note this does not define any "test_*" method, instead check_module()
    should be called by hand.
    """

    def check_module(self, mod):
        f = mod.f
        a = np.ones(100)
        b = np.ones(100)
        f(a, b)

    # For 2.x
    def runTest(self):
        raise NotImplementedError


def self_test():
    mod = sys.modules[__name__]
    _TestModule().check_module(mod)
