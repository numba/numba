"""
Tests for SSA reconstruction
"""
import sys
import copy
import logging

import numpy as np

from numba import njit
from numba.core import errors
from numba.tests.support import TestCase


_DEBUG = False

if _DEBUG:
    # Enable debug logger on SSA reconstruction
    ssa_logger = logging.getLogger("numba.core.ssa")
    ssa_logger.setLevel(level=logging.DEBUG)
    ssa_logger.addHandler(logging.StreamHandler(sys.stderr))


class TestSSA(TestCase):
    """
    Contains tests to help isolate problems in SSA
    """
    def check_func(self, func, *args):
        got = func(*copy.deepcopy(args))
        exp = func.py_func(*copy.deepcopy(args))
        self.assertEqual(got, exp)

    def test_argument_name_reused(self):
        @njit
        def foo(x):
            x += 1
            return x

        self.check_func(foo, 123)

    def test_if_else_redefine(self):
        @njit
        def foo(x, y):
            z = x * y
            if x < y:
                z = x
            else:
                z = y
            return z

        self.check_func(foo, 3, 2)
        self.check_func(foo, 2, 3)

    def test_sum_loop(self):
        @njit
        def foo(n):
            c = 0
            for i in range(n):
                c += i
            return c

        self.check_func(foo, 0)
        self.check_func(foo, 10)

    def test_sum_loop_2vars(self):
        @njit
        def foo(n):
            c = 0
            d = n
            for i in range(n):
                c += i
                d += n
            return c, d

        self.check_func(foo, 0)
        self.check_func(foo, 10)

    def test_sum_2d_loop(self):
        @njit
        def foo(n):
            c = 0
            for i in range(n):
                for j in range(n):
                    c += j
                c += i
            return c

        self.check_func(foo, 0)
        self.check_func(foo, 10)

    def test_undefined_var(self):
        @njit
        def foo(n):
            if n:
                if n > 0:
                    c = 0
                return c
            else:
                # variable c is not defined in this branch
                c += 1
                return c

        with self.assertRaises(errors.NotDefinedError) as raises:
            self.check_func(foo, 1)
        self.assertEqual(raises.exception.name, "c")

    def test_phi_propagation(self):
        @njit
        def foo(actions):
            n = 1

            i = 0
            ct = 0
            while n > 0 and i < len(actions):
                n -= 1

                while actions[i]:
                    if actions[i]:
                        if actions[i]:
                            n += 10
                        actions[i] -= 1
                    else:
                        if actions[i]:
                            n += 20
                        actions[i] += 1

                    ct += n
                ct += n
            return ct, n

        self.check_func(foo, np.array([1, 2]))
