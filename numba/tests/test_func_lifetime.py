
from __future__ import print_function, absolute_import

import gc
import weakref

from numba import unittest_support as unittest
from numba.utils import IS_PY3
from numba import jit, types
from .support import TestCase


def global_func(x):
    return x + 1


class TestFuncLifetime(TestCase):
    """
    Test the lifetime of compiled function objects.
    """
    # NOTE: there's a test for closure lifetime in test_closure

    def check_local_func_lifetime(self, **jitargs):
        def f(x):
            return x + 1

        c_f = jit(**jitargs)(f)
        self.assertPreciseEqual(c_f(1), 2)

        refs = [weakref.ref(obj) for obj in (f, c_f)]
        obj = f = c_f = None
        gc.collect()
        self.assertEqual([wr() for wr in refs], [None] * len(refs))

    def test_local_func_lifetime(self):
        self.check_local_func_lifetime(forceobj=True)

    def test_local_func_lifetime_npm(self):
        self.check_local_func_lifetime(nopython=True)

    def check_global_func_lifetime(self, **jitargs):
        c_f = jit(**jitargs)(global_func)
        self.assertPreciseEqual(c_f(1), 2)

        wr = weakref.ref(c_f)
        c_f = None
        gc.collect()
        self.assertIs(wr(), None)

    def test_global_func_lifetime(self):
        self.check_global_func_lifetime(forceobj=True)

    def test_global_func_lifetime_npm(self):
        self.check_global_func_lifetime(nopython=True)


if __name__ == '__main__':
    unittest.main()
