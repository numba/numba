from __future__ import print_function

import random

import numpy as np

from .support import TestCase
from numba import njit
from numba.unsafe.tuple import tuple_setitem
from numba.unsafe.ndarray import to_fixed_tuple
from numba.errors import TypingError


class TestTupleIntrinsic(TestCase):
    """Tests for numba.unsafe.tuple
    """
    def test_tuple_setitem(self):
        @njit
        def foo(tup, idxs, vals):
            out_tup = tup
            for i, v in zip(idxs, vals):
                out_tup = tuple_setitem(out_tup, i, v)
            return tup, out_tup

        random.seed(123)
        for _ in range(20):
            # Random data
            n = random.randint(1, 10)
            tup = tuple([random.randint(0, n) for i in range(n)])
            vals = tuple([random.randint(10, 20) for i in range(n)])
            idxs = list(range(len(vals)))
            random.shuffle(idxs)
            idxs = tuple(idxs)
            # Expect
            expect_tup = tuple(tup)
            expect_out = np.asarray(expect_tup)
            expect_out[np.asarray(idxs)] = vals
            # Got
            got_tup, got_out = foo(tup, idxs, vals)
            # Check
            self.assertEqual(got_tup, expect_tup)
            self.assertEqual(got_out, tuple(expect_out))


class TestNdarrayIntrinsic(TestCase):
    """Tests for numba.unsafe.ndarray
    """
    def test_to_fixed_tuple(self):
        const = 3

        @njit
        def foo(array):
            a = to_fixed_tuple(array, length=1)
            b = to_fixed_tuple(array, 2)
            c = to_fixed_tuple(array, const)
            d = to_fixed_tuple(array, 0)
            return a, b, c, d

        np.random.seed(123)
        for _ in range(10):
            # Random data
            arr = np.random.random(3)
            # Run
            a, b, c, d = foo(arr)
            # Check
            self.assertEqual(a, tuple(arr[:1]))
            self.assertEqual(b, tuple(arr[:2]))
            self.assertEqual(c, tuple(arr[:3]))
            self.assertEqual(d, ())

        # Check error with ndim!=1
        with self.assertRaises(TypingError) as raises:
            foo(np.random.random((1, 2)))
        self.assertIn("Not supported on array.ndim=2",
                      str(raises.exception))

        # Check error with non-constant length
        @njit
        def tuple_with_length(array, length):
            return to_fixed_tuple(array, length)

        with self.assertRaises(TypingError) as raises:
            tuple_with_length(np.random.random(3), 1)
        expectmsg = "*length* argument must be a constant"
        self.assertIn(expectmsg, str(raises.exception))
