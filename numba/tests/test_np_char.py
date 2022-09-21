"""Test string operations of the numpy.char module."""

from itertools import product
from numba import jit
from numba.core.errors import TypingError
from numba.np import char
from numba.tests.support import TestCase
from sys import maxunicode

import numpy as np
import unittest


# -----------------------------------------------------------------------------
# Support Functions

def _pack_arguments(main_args: (list, tuple), args: (list, tuple)):
    """Generate combinations of arguments for a list of main arguments"""
    arg_product = tuple(product(*args))
    for a in main_args:
        for args in arg_product:
            yield *a, *args


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
        length = 500
        # 500 UTF-32 strings of length 1 to 200 in range(1, sys.maxunicode)
        np.random.seed(42)
        q = np.array([''.join([chr(np.random.randint(maxunicode))
                               for _ in range(np.random.randint(1, 200))])
                      for _ in range(length)])
        # 500 ASCII strings of length 0 to 50
        r = np.array([''.join([chr(np.random.randint(1, 127))
                               for _ in range(np.random.randint(0, 50))])
                      for _ in range(length)])
        s: tuple = (np.array(['abc', 'def'] * length),
                    np.array(['cba', 'fed'] * length))
        t: tuple = (np.array(['ab', 'bc'] * length),
                    np.array(['bc', 'ab'] * length))
        u: tuple = (np.array(['ba', 'cb'] * length),
                    np.array(['cb', 'ba'] * length))
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

    def test_comparison_operators(self):

        def check_output(pyfunc_, cfunc_, x1, x2):
            expected = pyfunc_(x1, x2)
            got = cfunc_(x1, x2)
            self.assertPreciseEqual(expected, got)

        def check_exceptions(cfunc_, x1, x2):
            with self.assertRaisesRegex(ValueError, 'shape mismatch'):
                cfunc_(x1, np.empty_like(x1[:1]))
                cfunc_(x1[:10], x2[:5])

            with self.assertRaisesRegex(TypingError, 'non-string arrays'):
                cfunc_(x1[:5], None)
                cfunc_('abc', 123)

            with self.assertRaisesRegex(TypingError, 'NotImplemented'):
                cfunc_(x1.astype('U'), x1.astype('S'))
                cfunc_('abc', b'abc')

        pyfuncs = (np_char_equal, np_char_not_equal,
                   np_char_greater_equal, np_char_greater,
                   np_char_less_equal, np_char_less)

        args = self.string_args[0]
        for pyfunc in pyfuncs:
            cfunc = jit(nopython=True)(pyfunc)
            check_exceptions(cfunc, *args)

        for pyfunc in pyfuncs:
            cfunc = jit(nopython=True)(pyfunc)
            for args in self.byte_args:
                check_output(pyfunc, cfunc, *args)
                check_output(pyfunc, cfunc, *args[::-1])

            for args in self.string_args:
                check_output(pyfunc, cfunc, *args)
                check_output(pyfunc, cfunc, *args[::-1])

        byte_args = list(_pack_arguments(self.byte_args,
                                         [('==', '!=', '>=', '>', '<', '<='),
                                          (True, False)]))

        string_args = list(_pack_arguments(self.string_args,
                                           [('==', '!=', '>=', '>', '<', '<='),
                                            (True, False)]))

        def check_compare_chararrays(a1, a2, cmp, rstrip):
            pyfunc_ = np_char_compare_chararrays
            cfunc_ = jit(nopython=True)(pyfunc_)
            expected = pyfunc_(a1, a2, cmp, rstrip)
            got = cfunc_(a1, a2, cmp, rstrip)
            self.assertPreciseEqual(expected, got)

        for args in byte_args:
            check_compare_chararrays(*args)

        for args in string_args:
            check_compare_chararrays(*args)


TestComparisonOperators.set_arguments()


if __name__ == '__main__':
    unittest.main()
