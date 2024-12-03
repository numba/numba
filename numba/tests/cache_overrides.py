"""
This file will be copied to a temporary directory in order to
exercise overriding of caching compiled functions.

See test_caching.py (TestCacheOverrides class)
"""

import sys

from numba import jit, njit, cfunc, double as ndouble
from numba.tests.support import TestCase


@jit(cache=True)
def add_jit_cached(a, b):
    return a + b


@jit(cache=False)
def add_jit_notcached(a, b):
    return a + b


@njit(cache=True)
def add_njit_cached(a, b):
    return a + b


@njit(cache=False)
def add_njit_notcached(a, b):
    return a + b


@cfunc(ndouble(ndouble, ndouble), cache=True)
def add_cfunc_cached(a, b):
    return a + b


@cfunc(ndouble(ndouble, ndouble), cache=False)
def add_cfunc_notcached(a, b):
    return a + b


class _TestModule(TestCase):
    """
    Tests for functionality of this module's functions.
    Note this does not define any "test_*" method, instead check_module()
    should be called by hand.
    """

    def check_module(self, mod):
        f = mod.add_jit_cached
        self.assertPreciseEqual(f(2.0, 3.0), 5.0)
        f = mod.add_jit_notcached
        self.assertPreciseEqual(f(-2.0, 3.0), 1.0)

        f = mod.add_njit_cached
        self.assertPreciseEqual(f(2.0, 3.0), 5.0)
        f = mod.add_njit_notcached
        self.assertPreciseEqual(f(-2.0, 3.0), 1.0)

        f = mod.add_cfunc_cached
        self.assertPreciseEqual(f.ctypes(2.0, 3.0), 5.0)
        f = mod.add_cfunc_notcached
        self.assertPreciseEqual(f.ctypes(-2.0, 3.0), 1.0)


def self_test():
    mod = sys.modules[__name__]
    _TestModule().check_module(mod)
