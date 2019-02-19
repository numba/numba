from __future__ import print_function

import random
import numpy as np

from .support import TestCase, captured_stdout
from numba import njit, types
from numba.unsafe.tuple import tuple_setitem
from numba.unsafe.ndarray import to_fixed_tuple, empty_inferred
from numba.unsafe.bytes import memcpy_region
from numba.unsafe.refcount import dump_refcount
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

    def test_issue_3586_variant1(self):
        @njit
        def func():
            S = empty_inferred((10,))
            a = 1.1
            for i in range(len(S)):
                S[i] = a + 2
            return S

        got = func()
        expect = np.asarray([3.1] * 10)
        np.testing.assert_array_equal(got, expect)

    def test_issue_3586_variant2(self):
        @njit
        def func():
            S = empty_inferred((10,))
            a = 1.1
            for i in range(S.size):
                S[i] = a + 2
            return S

        got = func()
        expect = np.asarray([3.1] * 10)
        np.testing.assert_array_equal(got, expect)


class TestBytesIntrinsic(TestCase):
    """Tests for numba.unsafe.bytes
    """
    def test_memcpy_region(self):
        @njit
        def foo(dst, dst_index, src, src_index, nbytes):
            # last arg is assume 1 byte alignment
            memcpy_region(dst.ctypes.data, dst_index,
                          src.ctypes.data, src_index, nbytes, 1)

        d = np.zeros(10, dtype=np.int8)
        s = np.arange(10, dtype=np.int8)

        # copy s[1:6] to d[4:9]
        foo(d, 4, s, 1, 5)

        expected = [0, 0, 0, 0, 1, 2, 3, 4, 5, 0]
        np.testing.assert_array_equal(d, expected)


class TestRefCount(TestCase):
    def test_dump_refcount(self):
        @njit
        def use_dump_refcount():
            a = np.ones(10)
            b = (a, a)
            dump_refcount(a)
            dump_refcount(b)

        # Capture output to sys.stdout
        with captured_stdout() as stream:
            use_dump_refcount()

        output = stream.getvalue()
        # Check that it printed
        pat = "dump refct of {}"
        aryty = types.float64[::1]
        tupty = types.Tuple.from_types([aryty] * 2)
        self.assertIn(pat.format(aryty), output)
        self.assertIn(pat.format(tupty), output)
