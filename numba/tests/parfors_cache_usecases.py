import sys

import numpy as np

from numba import njit
from numba.tests.support import TestCase
from numba.npyufunc.parallel import _launch_threads

_launch_threads()  # FIXME


@njit(parallel=True, cache=True)
def arrayexprs(arr):
    return arr / arr.sum()


class _TestModule(TestCase):
    """
    Tests for functionality of this module's functions.
    Note this does not define any "test_*" method, instead check_module()
    should be called by hand.
    """

    def check_module(self, mod):
        arr = np.ones(2)
        self.assertPreciseEqual(
            mod.arrayexprs(arr), mod.arrayexprs.py_func(arr),
        )

    # For 2.x
    def runTest(self):
        raise NotImplementedError


def self_test():
    mod = sys.modules[__name__]
    _TestModule().check_module(mod)
