"""Test string operations of the numpy.char module."""

from itertools import product
from numba import jit
from numba.core.errors import TypingError
from numba.np.char import np
from numba.tests.support import TestCase
from sys import maxunicode

import unittest


# -----------------------------------------------------------------------------
# Support Functions

def _pack_arguments(main_args: (list, tuple), args: (list, tuple)):
    """Generate combinations of arguments for a list of main arguments"""
    arg_product = tuple(product(*args))
    for a in main_args:
        for args in arg_product:
            yield (*a, *args)


def _arguments_as_bytes(args: (list, tuple)):
    """Yield byte counterparts of string arguments, given a list of arguments"""
    for pair in args:
        as_bytes = []
        for arg in pair:
            if isinstance(arg, np.ndarray):
                as_bytes.append(arg.astype('S'))
            elif isinstance(arg, str):
                as_bytes.append(bytes(arg, 'UTF-8'))
            else:
                as_bytes.append(arg)
        yield as_bytes


# -----------------------------------------------------------------------------
# Comparison Operators

def np_char_equal(x1, x2):
    return np.char.equal(x1, x2)


def np_char_not_equal(x1, x2):
    return np.char.not_equal(x1, x2)


def np_char_greater(x1, x2):
    return np.char.greater(x1, x2)


def np_char_greater_equal(x1, x2):
    return np.char.greater_equal(x1, x2)


def np_char_less(x1, x2):
    return np.char.less(x1, x2)


def np_char_less_equal(x1, x2):
    return np.char.less_equal(x1, x2)


def np_char_compare_chararrays(a1, a2, cmp, rstrip):
    return np.char.compare_chararrays(a1, a2, cmp, rstrip)


class TestComparisonOperators(TestCase):
    """Test comparison operators of the numpy.char module."""

    byte_args, string_args = [], []

    @classmethod
    def set_arguments(cls):
        length = 300
        np.random.seed(42)
        # 300 UTF-32 strings of length 1 to 200 in range(1, sys.maxunicode)
        # Python 3.7 can not decode unicode in range(55296, 57344)
        q = np.array([''.join([chr(np.random.randint(1, 55295)) if i % 2
                               else chr(np.random.randint(57345, maxunicode))
                               for i in range(np.random.randint(1, 200))])
                      for _ in range(length)])
        # 300 ASCII strings of length 0 to 50
        r = np.array([''.join([chr(np.random.randint(1, 127))
                               for _ in range(np.random.randint(0, 50))])
                      for _ in range(length)])
        s: tuple = (np.array(['abc', 'def'] * 2),
                    np.array(['cba', 'fed'] * 2))
        t: tuple = (np.array(['ab', 'bc'] * 2),
                    np.array(['bc', 'ab'] * 2))
        u: tuple = (np.array(['ba', 'cb'] * 2),
                    np.array(['cb', 'ba'] * 2))
        v = np.random.choice(['abcd', 'abc', 'abcde'], length)

        # Whitespace to end of strings & single ASCII characters in range(0, 33)
        w = [chr(i) for i in range(33)]
        x = np.concatenate([w, np.char.add(r, np.random.choice(w, length))])

        # Single ASCII characters
        c = np.random.choice([chr(__i) for __i in range(128)], length)

        arrays = [
            s, t, u,
            (c, np.random.choice(c, c.size)),
            (v, np.random.choice(v, v.size)),
            (x, x),
            (x, np.random.choice(x, x.size)),
            (x, 'abcdefg'), (x, 'abcdefg' * 50),
            ('abc', 'abc'), ('abc', 'abd'), ('abc', 'abb'),
            ('abc', 'abc' * 100), ('ab', 'ba'),
        ]

        # Character buffers of different length
        arrays += [
            (x, x.astype('U200')),
            (s[0].astype('U20'), s[1].astype('U40')),
            (x.astype('U60'), x.astype('U61')),
            (np.array('hello' * 100, dtype='U200'),
             np.array('hello' * 100, dtype='U100'))
        ]

        # UTF-32
        arrays += [
            (q, np.random.choice(q)),
            (q, np.random.choice(q, len(q))),
            (q, np.char.add(q, np.random.choice(w, len(q))))
        ]

        byte_args = list(_arguments_as_bytes(arrays[:-3]))
        string_args = arrays

        setattr(cls, 'byte_args', byte_args)
        setattr(cls, 'string_args', string_args)

    def test_comparisons(self):

        def check_output(pyfunc_, cfunc_, x1, x2):
            expected = pyfunc_(x1, x2)
            got = cfunc_(x1, x2)
            self.assertPreciseEqual(expected, got)

        def check_shape_exception(cfunc_, x1):
            error_msg = ".*shape mismatch: objects cannot be broadcast to.*"
            with self.assertRaisesRegex(ValueError, error_msg):
                cfunc_(x1, x1[:2])

        def check_comparison_exception(cfunc_, x1):
            error_msg = ".*comparison of non-string arrays.*"
            accepted_errors = (TypingError, TypeError)
            with self.assertRaisesRegex(accepted_errors, error_msg):
                cfunc_(x1, None)
            with self.assertRaisesRegex(accepted_errors, error_msg):
                cfunc_(x1, 123)

        def check_notimplemented_exception(cfunc_, x1):
            error_msg = ".*NotImplemented.*"
            accepted_errors = (TypingError, NotImplementedError)
            with self.assertRaisesRegex(accepted_errors, error_msg):
                cfunc_(x1.astype('S'), x1.astype('U'))
            with self.assertRaisesRegex(accepted_errors, error_msg):
                cfunc_('abc', b'abc')

        pyfuncs = (np_char_equal, np_char_not_equal,
                   np_char_greater_equal, np_char_greater,
                   np_char_less_equal, np_char_less)

        arg = np.array(['abc', 'def', 'hij'], 'S')
        for pyfunc in pyfuncs:
            cfunc = jit(nopython=True)(pyfunc)
            check_shape_exception(cfunc, arg)
            check_comparison_exception(cfunc, arg)
            check_notimplemented_exception(cfunc, arg)

        for pyfunc in pyfuncs:
            cfunc = jit(nopython=True)(pyfunc)
            for args in self.byte_args:
                check_output(pyfunc, cfunc, *args)
                check_output(pyfunc, cfunc, *args[::-1])

            for args in self.string_args:
                check_output(pyfunc, cfunc, *args)
                check_output(pyfunc, cfunc, *args[::-1])

    def test_compare_chararrays(self):

        def check_output(a1, a2, cmp, rstrip):
            expected = pyfunc(a1, a2, cmp, rstrip)
            got = cfunc(a1, a2, cmp, rstrip)
            self.assertPreciseEqual(expected, got)

        def check_cmp_exception():
            cmp = 123
            error_msg = ".*a bytes-like object is required.*"
            accepted_errors = (TypingError, TypeError)
            with self.assertRaisesRegex(accepted_errors, error_msg):
                cfunc('abc', 'abc', cmp, True)

        pyfunc = np_char_compare_chararrays
        cfunc = jit(nopython=True)(pyfunc)

        byte_args = _pack_arguments(self.byte_args[:1],
                                    [('==', '!=', '>=', '>', '<', '<='),
                                     (True, False)])

        string_args = _pack_arguments(self.string_args[:1],
                                      [('==', '!=', '>=', '>', '<', '<='),
                                       (True, False)])

        check_cmp_exception()

        for args in byte_args:
            check_output(*args)

        for args in string_args:
            check_output(*args)


TestComparisonOperators.set_arguments()


if __name__ == '__main__':
    unittest.main()
