import random
import unittest

import numpy as np

from numba import literal_unroll, njit
from numba.core import types
from numba.core.errors import TypingError
from numba.core.unsafe.bytes import memcpy_region
from numba.core.unsafe.refcount import dump_refcount
from numba.cpython.unsafe.numbers import leading_zeros, trailing_zeros
from numba.cpython.unsafe.tuple import tuple_setitem
from numba.np.unsafe.ndarray import empty_inferred, to_fixed_tuple
from numba.tests.support import TestCase, captured_stdout


class TestTupleIntrinsic(TestCase):
    """Tests for numba.unsafe.tuple
    """
    def test_tuple_setitem(self):
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
            @njit
            def foo(tup):
                out_tup = tup
                counter = 0
                for i in literal_unroll(idxs):
                    v = vals[counter]
                    out_tup = tuple_setitem(out_tup, i, v)
                    counter += 1
                return tup, out_tup

            got_tup, got_out = foo(tup)

            # Check
            self.assertEqual(got_tup, expect_tup)
            self.assertEqual(got_out, tuple(expect_out))

    def test_tuple_setitem_heterogeneous(self):
        test_case = [1, 'a', np.zeros(2)]
        tup = tuple(test_case)

        # Expect
        expected_case = test_case.copy()
        i = 1
        v = 'b'
        expected_case[i] = v
        expected = tuple(expected_case)

        # Got
        @njit
        def foo(tup, v):
            return tuple_setitem(tup, i, v)

        got_out = foo(tup, v)

        # Check
        self.assertEqual(got_out, expected)

        # Expect
        expected_case = test_case.copy()
        i = 2
        v = np.ones(2)
        expected_case[i] = v
        expected = tuple(expected_case)

        # Got
        @njit
        def foo(tup, v):
            return tuple_setitem(tup, i, v)

        got_out = foo(tup, v)

        # Check
        self.assertEqual(got_out, expected)

    def test_tuple_setitem_heterogeneous_invalid_inputs(self):
        test_case = [1, 'a', np.zeros(2)]
        tup = tuple(test_case)
        v = 'b'

        # Check non-tuple input
        i = 1

        @njit
        def foo(tup, v):
            return tuple_setitem(1, i, v)

        with self.assertRaises(TypingError) as raises:
            foo(tup, v)
        self.assertIn("must of type tuple, got type IntegerLiteral",
                      str(raises.exception))

        # Check negative index
        i = -1

        @njit
        def foo(tup, v):
            return tuple_setitem(tup, i, v)

        with self.assertRaises(TypingError) as raises:
            foo(tup, v)
        self.assertIn("index out of range", str(raises.exception))

        # Check index > length - 1
        i = 3

        @njit
        def foo(tup, v):
            return tuple_setitem(tup, i, v)

        with self.assertRaises(TypingError) as raises:
            foo(tup, v)
        self.assertIn("index out of range", str(raises.exception))

        # Check incompatible type of new value
        i = 2

        @njit
        def foo(tup, v):
            return tuple_setitem(tup, i, v)

        with self.assertRaises(TypingError) as raises:
            foo(tup, v)
        self.assertIn("must of type array(float64, 1d, C), got type "
                      "UnicodeType",
                      str(raises.exception))


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


class TestZeroCounts(TestCase):
    def test_zero_count(self):
        lz = njit(lambda x: leading_zeros(x))
        tz = njit(lambda x: trailing_zeros(x))

        evens = [2, 42, 126, 128]

        for T in types.unsigned_domain:
            self.assertTrue(tz(T(0)) == lz(T(0)) == T.bitwidth)
            for i in range(T.bitwidth):
                val = T(2 ** i)
                self.assertEqual(lz(val) + tz(val) + 1, T.bitwidth)
            for n in evens:
                self.assertGreater(tz(T(n)), 0)
                self.assertEqual(tz(T(n + 1)), 0)

        for T in types.signed_domain:
            self.assertTrue(tz(T(0)) == lz(T(0)) == T.bitwidth)
            for i in range(T.bitwidth - 1):
                val = T(2 ** i)
                self.assertEqual(lz(val) + tz(val) + 1, T.bitwidth)
                self.assertEqual(lz(-val), 0)
                self.assertEqual(tz(val), tz(-val))
            for n in evens:
                self.assertGreater(tz(T(n)), 0)
                self.assertEqual(tz(T(n + 1)), 0)

    def check_error_msg(self, func):
        cfunc = njit(lambda *x: func(*x))
        func_name = func._name

        unsupported_types = filter(
            lambda x: not isinstance(x, types.Integer), types.number_domain
        )
        for typ in unsupported_types:
            with self.assertRaises(TypingError) as e:
                cfunc(typ(2))
            self.assertIn(
                "{} is only defined for integers, but passed value was '{}'."
                .format(func_name, typ),
                str(e.exception),
            )

        # Testing w/ too many arguments
        arg_cases = [(1, 2), ()]
        for args in arg_cases:
            with self.assertRaises(TypingError) as e:
                cfunc(*args)
            self.assertIn(
                "Invalid use of Function({})".format(str(func)),
                str(e.exception)
            )

    def test_trailing_zeros_error(self):
        self.check_error_msg(trailing_zeros)

    def test_leading_zeros_error(self):
        self.check_error_msg(leading_zeros)


if __name__ == '__main__':
    unittest.main()
