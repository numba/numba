"""
This file will be copied to a temporary directory in order to
exercise caching compiled jitclass methods.

See test_caching.py
"""
import sys

import numba
from numba.experimental.jitclass import jitclass, jitmethod
from numba.tests.support import TestCase


spec = [
    ("a", numba.int64),
    ("b", numba.int64),
    ("c", numba.int64),
    ("total", numba.int64),
]


@jitclass(spec=spec, cache=True)
class MyJitClass:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        self.total = a + b + c

    def cached_undecorated_function(self, d):
        return self.a + d

    @jitmethod
    def cached_decorated_function_inherits_cache_setting(self, d):
        return self.b + d

    @jitmethod(cache=True)
    def cached_decorated_function_sets_cache_to_true(self, d):
        return self.c + d

    @jitmethod(cache=False)
    def uncached_function(self, d):
        return self.total + d


class _TestModule(TestCase):
    """
    Tests for functionality of this module's class.
    Note this does not define any "test_*" method, instead check_module()
    should be called by hand.
    """

    def check_module(self, mod):
        c: MyJitClass = mod.MyJitClass(1, 2, 3)
        self.assertPreciseEqual(c.a, 1)
        self.assertPreciseEqual(c.b, 2)
        self.assertPreciseEqual(c.c, 3)
        self.assertPreciseEqual(
            c.cached_undecorated_function(4), 5
        )
        self.assertPreciseEqual(
            c.cached_decorated_function_inherits_cache_setting(5), 7
        )
        self.assertPreciseEqual(
            c.cached_decorated_function_sets_cache_to_true(6), 9
        )
        self.assertPreciseEqual(c.uncached_function(7), 13)


def self_test():
    mod = sys.modules[__name__]
    _TestModule().check_module(mod)
