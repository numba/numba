"""
Tests for practical lowering specific errors.
"""


from __future__ import print_function

import numpy as np
from numba import njit

from .support import MemoryLeakMixin, TestCase


class TestLowering(MemoryLeakMixin, TestCase):
    def test_issue4156_loop_vars_leak(self):
        """Test issues with zero-filling of refct'ed variables inside loops.

        Before the fix, the in-loop variables are always zero-filled at their
        definition location. As a result, their state from the previous
        iteration is erased. No decref is applied. To fix this, the
        zero-filling must only happen once after the alloca at the function
        entry block. The loop variables are technically defined once per
        function (one alloca per definition per function), but semantically
        defined once per assignment. Semantically, their lifetime stop only
        when the variable is re-assigned or when the function ends.
        """
        @njit
        def udt(N):
            sum_vec = np.zeros(3)
            for n in range(N):
                if n >= 0:
                    # `vec` would leak without the fix.
                    vec = np.ones(1)
                if n >= 0:
                    sum_vec += vec[0]

            return sum_vec

        got = udt(4)
        expect = udt.py_func(4)
        self.assertPreciseEqual(got, expect)

    def test_issue4156_loop_vars_leak_variant1(self):
        """Variant of test_issue4156_loop_vars_leak.

        Adding an outer loop.
        """
        @njit
        def udt(N):
            sum_vec = np.zeros(3)
            for x in range(N):
                for y in range(N):
                    n = x + y
                    if n >= 0:
                        # `vec` would leak without the fix.
                        vec = np.ones(1)
                    if n >= 0:
                        sum_vec += vec[0]

            return sum_vec

        got = udt(4)
        expect = udt.py_func(4)
        self.assertPreciseEqual(got, expect)

    def test_issue4156_loop_vars_leak_variant2(self):
        """Variant of test_issue4156_loop_vars_leak.

        Adding deeper outer loop.
        """
        @njit
        def udt(N):
            sum_vec = np.zeros(3)
            for z in range(N):
                for x in range(N):
                    for y in range(N):
                        n = x + y + z
                        if n >= 0:
                            # `vec` would leak without the fix.
                            vec = np.ones(1)
                        if n >= 0:
                            sum_vec += vec[0]

            return sum_vec

        got = udt(4)
        expect = udt.py_func(4)
        self.assertPreciseEqual(got, expect)

    def test_issue4156_loop_vars_leak_variant3(self):
        """Variant of test_issue4156_loop_vars_leak.

        Adding inner loop around allocation
        """
        @njit
        def udt(N):
            sum_vec = np.zeros(3)
            for z in range(N):
                for x in range(N):
                    n = x + z
                    if n >= 0:
                        for y in range(N):
                            # `vec` would leak without the fix.
                            vec = np.ones(y)
                    if n >= 0:
                        sum_vec += vec[0]

            return sum_vec

        got = udt(4)
        expect = udt.py_func(4)
        self.assertPreciseEqual(got, expect)

    def test_issue4156_loop_vars_leak_variant4(self):
        """Variant of test_issue4156_loop_vars_leak.

        Interleaves loops and allocations
        """
        @njit
        def udt(N):
            sum_vec = 0

            for n in range(N):
                vec = np.zeros(7)
                for n in range(N):
                    z = np.zeros(7)
                sum_vec += vec[0] + z[0]

            return sum_vec

        got = udt(4)
        expect = udt.py_func(4)
        self.assertPreciseEqual(got, expect)
