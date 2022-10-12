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
    arg_product = product(*args)
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
        length = 100
        np.random.seed(42)
        # 100 ASCII strings of length 0 to 50
        s = np.array([''.join([chr(np.random.randint(1, 127))
                               for _ in range(np.random.randint(0, 50))])
                      for _ in range(length)])
        # 100 UTF-32 strings of length 1 to 200 in range(1, sys.maxunicode)
        # Python 3.7 can not decode unicode in range(55296, 57344)
        u = np.array([''.join([chr(np.random.randint(1, 55295)) if i % 2
                               else chr(np.random.randint(57345, maxunicode))
                               for i in range(np.random.randint(1, 200))])
                      for _ in range(length)])
        # Whitespace to end of strings & single ASCII characters in range(0, 33)
        w = [chr(i) for i in range(33)]
        x = np.concatenate([w, np.char.add(s, np.random.choice(w, length))])

        # Single ASCII characters
        c = np.random.choice([chr(i) for i in range(128)], length)

        generics = [
            (c, np.random.choice(c, c.size)),
            (x, np.random.choice(x, x.size)),
            (x[:2], x[:2]),
        ]

        # Scalar Comparisons
        scalars = [
            (x, 'abcd ' * 20),
            ('abc', 'abc '), ('abc', 'abc' * 2),
            ('abc', 'abd'), ('abc', 'abb'), ('ab', 'ba'),
        ]

        # Character buffers of different length
        buffers = [
            (s[:1].astype('U20'), s[:1].astype('U40')),
            (x[:5].astype('U60'), x[:5].astype('U61')),
            (x[:5], x[:5].astype('U100')),
            (np.array('hello ' * 5, dtype='U30'),
             np.array('hello ' * 10, dtype='U60')),
        ]

        # UTF-32
        utf32 = [
            (u, np.random.choice(u)),
            (u, np.random.choice(u, len(u))),
            (u, np.char.add(u, np.random.choice(w, len(u))))
        ]

        byte_args = generics + scalars + buffers
        string_args = byte_args + utf32

        setattr(cls, 'byte_args', list(_arguments_as_bytes(byte_args)))
        setattr(cls, 'string_args', string_args)

    def test_comparisons(self):

        pyfuncs = (np_char_equal, np_char_not_equal,
                   np_char_greater_equal, np_char_greater,
                   np_char_less_equal, np_char_less)

        def check_output(pyfunc_, cfunc_, x1, x2):
            expected = pyfunc_(x1, x2)
            got = cfunc_(x1, x2)
            self.assertPreciseEqual(expected, got)

        def check_shape_exception(cfunc_, x1):
            error_msg = ".*shape mismatch: objects cannot be broadcast to.*"
            with self.assertRaisesRegex(ValueError, error_msg):
                cfunc_(x1, x1[:2])

        def check_comparison_exception(cfunc_, x1):
            accepted_errors = (TypingError, TypeError)
            error_msg = ".*comparison of non-string arrays.*"
            with self.assertRaisesRegex(accepted_errors, error_msg):
                cfunc_(x1, None)
            with self.assertRaisesRegex(accepted_errors, error_msg):
                cfunc_(x1, 123)

        def check_notimplemented_exception(cfunc_, x1):
            accepted_errors = (TypingError, NotImplementedError)
            error_msg = ".*NotImplemented.*"
            with self.assertRaisesRegex(accepted_errors, error_msg):
                cfunc_(x1.astype('S'), x1.astype('U'))
            with self.assertRaisesRegex(accepted_errors, error_msg):
                cfunc_('abc', b'abc')

        arg = np.array(['abc', 'def', 'hij'], 'S')
        for pyfunc in pyfuncs:
            cfunc = jit(nopython=True)(pyfunc)
            check_shape_exception(cfunc, arg)
            check_comparison_exception(cfunc, arg)
            check_notimplemented_exception(cfunc, arg)

            for args in self.byte_args:
                check_output(pyfunc, cfunc, *args)
                check_output(pyfunc, cfunc, *args[::-1])

            for args in self.string_args:
                check_output(pyfunc, cfunc, *args)
                check_output(pyfunc, cfunc, *args[::-1])

    def test_compare_chararrays(self):

        pyfunc = np_char_compare_chararrays
        cfunc = jit(nopython=True)(pyfunc)

        def check_output(a1, a2, cmp, rstrip):
            expected = pyfunc(a1, a2, cmp, rstrip)
            got = cfunc(a1, a2, cmp, rstrip)
            self.assertPreciseEqual(expected, got)

        def check_cmp_exception():
            cmp = 123
            accepted_errors = (TypingError, TypeError)
            error_msg = ".*a bytes-like object is required.*"
            with self.assertRaisesRegex(accepted_errors, error_msg):
                cfunc('abc', 'abc', cmp, True)

        check_cmp_exception()

        byte_args = _pack_arguments(self.byte_args[:2],
                                    [('==', '!=', '>=', '>', '<', '<='),
                                     (True, False)])
        string_args = _pack_arguments(self.string_args[:2],
                                      [('==', '!=', '>=', '>', '<', '<='),
                                       (True, False)])
        for args in byte_args:
            check_output(*args)

        for args in string_args:
            check_output(*args)


# -----------------------------------------------------------------------------
# String Information

# Occurrence Methods
# ******************
def np_char_count(a, sub, start=0, end=None):
    return np.char.count(a, sub, start, end)


def np_char_endswith(a, suffix, start=0, end=None):
    return np.char.endswith(a, suffix, start, end)


def np_char_startswith(a, prefix, start=0, end=None):
    return np.char.startswith(a, prefix, start, end)


def np_char_find(a, sub, start=0, end=None):
    return np.char.find(a, sub, start, end)


def np_char_rfind(a, sub, start=0, end=None):
    return np.char.rfind(a, sub, start, end)


def np_char_index(a, sub, start=0, end=None):
    return np.char.index(a, sub, start, end)


def np_char_rindex(a, sub, start=0, end=None):
    return np.char.rindex(a, sub, start, end)


# Property Methods
# ****************
def np_char_str_len(a):
    return np.char.str_len(a)


def np_char_isalpha(a):
    return np.char.isalpha(a)


def np_char_isalnum(a):
    return np.char.isalnum(a)


def np_char_isspace(a):
    return np.char.isspace(a)


def np_char_isdecimal(a):
    return np.char.isdecimal(a)


def np_char_isdigit(a):
    return np.char.isdigit(a)


def np_char_isnumeric(a):
    return np.char.isnumeric(a)


def np_char_istitle(a):
    return np.char.istitle(a)


def np_char_isupper(a):
    return np.char.isupper(a)


def np_char_islower(a):
    return np.char.islower(a)


class TestStringInformation(TestCase):
    """Test string information methods of the numpy.char module."""

    byte_args, string_args, property_args = [], [], []

    @classmethod
    def set_arguments(cls):
        length = 100
        np.random.seed(42)

        # Whitespace
        w = [9, 10, 11, 12, 13, 28, 29, 30, 31, 32]

        # ASCII word pairs
        a = np.array(['aAaAaA', 'abBABba', 'AbbAbbbbAbb', '  aA  ',
                      'Aa aA ', 'aa aa', 'AA', 'A1 1A', '2a', '33', 'Aa-aa'])
        p = np.array([chr(np.random.choice(w))
                     .join([''.join([chr(np.random.randint(48, 127))
                                     for _ in range(3)]) for _ in range(2)])
                      for _ in range(length)])

        # ASCII strings of length 0 to 50
        s = np.array([''.join([chr(np.random.randint(1, 127))
                               for _ in range(np.random.randint(0, 50))])
                      for _ in range(length)])

        # UTF-32 strings of length 1 to 200 in range(1, sys.maxunicode)
        # Python 3.7 can not decode unicode in range(55296, 57344)
        u = np.array([''.join([chr(np.random.randint(1, 55295)) if i % 2
                               else chr(np.random.randint(57345, maxunicode))
                               for i in range(np.random.randint(1, 200))])
                      for _ in range(length)])
        # Single ASCII characters
        c = np.random.choice([chr(i) for i in range(128)], length)

        generics = [
            (c, np.random.choice(c, c.size)),
            (s, np.random.choice(s, s.size)),
            (np.random.choice(s), np.array(['a', 'b', 'c', '']), 'U1')
        ]

        # Scalar Comparisons
        scalars = [
            (a, 'aA'), (a, 'Abb'),
            ('abc' * 2, 'abc'), ('abc', ''), ('', 'abc')
        ]

        # Character buffers of different length
        buffers = [
            (a.astype('U35'), np.array('A', 'U40')),
            (a, np.array('', 'U10'))
        ]

        # UTF-32
        utf32 = [
            (u, np.random.choice(u)),
            (u, np.random.choice(u, u.size)),
        ]

        # String Property Arguments
        property_args = [(a, ), (c, ), (p, ), (s, )]

        byte_args = generics + scalars + buffers
        string_args = byte_args + utf32

        setattr(cls, 'byte_args', list(_arguments_as_bytes(byte_args)))
        setattr(cls, 'string_args', string_args)
        setattr(cls, 'property_args', property_args)

    def test_string_occurrence(self):

        pyfuncs = (np_char_count,
                   np_char_endswith, np_char_startswith,
                   np_char_find, np_char_rfind,
                   np_char_index, np_char_rindex)

        def check_output(pyfunc_, cfunc_, x1, x2):
            expected = pyfunc_(x1, x2)
            got = cfunc_(x1, x2)
            self.assertPreciseEqual(expected, got)

        def check_slice_exception(cfunc_):
            accepted_errors = (TypingError, TypeError)
            error_msg = ".*slice indices must be integers or None.*"
            with self.assertRaisesRegex(accepted_errors, error_msg):
                cfunc_('abc', 'a', 0, 1.0)
            with self.assertRaisesRegex(accepted_errors, error_msg):
                cfunc_('abc', 'a', 'start', 2)

        def check_type_exception(cfunc_):
            accepted_errors = (TypingError, TypeError)
            error_msg = ".*must be str, not.*"
            with self.assertRaisesRegex(accepted_errors, error_msg):
                cfunc_('abc', b'abc')
            error_msg = ".*must be bytes, not.*"
            with self.assertRaisesRegex(accepted_errors, error_msg):
                cfunc_(b'abc', 'abc')

            error_msg = ".*string operation on non-string array.*"
            with self.assertRaisesRegex(accepted_errors, error_msg):
                cfunc_(123, 'abc')
            with self.assertRaisesRegex(accepted_errors, error_msg):
                cfunc_(None, 'abc')

        def test_index_exception(cfunc_):
            # A ValueError is raised when the substring argument
            # is unmatched for `index` and `rindex`.
            accepted_errors = ValueError
            error_msg = ".*substring not found.*"
            with self.assertRaisesRegex(accepted_errors, error_msg):
                cfunc_('abc', 'd')
            with self.assertRaisesRegex(accepted_errors, error_msg):
                cfunc_(np.array([b'abc']), np.array([b'b', b'c', b'd']))

        for pyfunc in pyfuncs:
            cfunc = jit(nopython=True)(pyfunc)
            check_type_exception(cfunc)
            check_slice_exception(cfunc)

            if cfunc.__name__ in ('np_char_index', 'np_char_rindex'):
                test_index_exception(cfunc)
                continue

            byte_args = _pack_arguments(self.byte_args,
                                        [(None, -2, 1, 2, -500, 500),
                                         (0, -2, 1, 2, -500, None)])
            string_args = _pack_arguments(self.string_args,
                                          [(None, -2, 1, 2, -500, 500),
                                           (0, -2, 1, 2, -500, None)])
            for args in byte_args:
                check_output(pyfunc, cfunc, *args)

            for args in string_args:
                check_output(pyfunc, cfunc, *args)

    def test_string_properties(self):

        pyfuncs = (np_char_str_len,
                   np_char_isalpha, np_char_isalnum, np_char_isspace,
                   np_char_isdecimal, np_char_isdigit, np_char_isnumeric,
                   np_char_istitle, np_char_isupper, np_char_islower)

        def check_output(pyfunc_, cfunc_, x1, x2):
            expected = pyfunc_(x1, x2)
            got = cfunc_(x1, x2)
            self.assertPreciseEqual(expected, got)

        def check_type_exception(cfunc_):
            accepted_errors = (TypingError, TypeError)
            error_msg = ".*string operation on non-string array.*"
            with self.assertRaisesRegex(accepted_errors, error_msg):
                cfunc_(123)
            with self.assertRaisesRegex(accepted_errors, error_msg):
                cfunc_(None)

        def check_isnumeric_exception(cfunc_):
            # A TypeError with the following 'isnumeric' message
            # is raised with byte arguments for `isnumeric` and `isdecimal`.
            accepted_errors = (TypingError, TypeError)
            error_msg = ".*isnumeric is only available for Unicode strings.*"
            with self.assertRaisesRegex(accepted_errors, error_msg):
                cfunc_(b'10')
            with self.assertRaisesRegex(accepted_errors, error_msg):
                cfunc_(np.array([b'1', b'2'], dtype='S1'))

        for pyfunc in pyfuncs:
            cfunc = jit(nopython=True)(pyfunc)

            for args in self.property_args:
                check_output(pyfunc, cfunc, *args)

            if cfunc.__name__ in ('np_char_isdecimal', 'np_char_isnumeric'):
                check_isnumeric_exception(cfunc)
                continue
            check_type_exception(cfunc)

            for args in _arguments_as_bytes(self.property_args):
                check_output(pyfunc, cfunc, *args)


if __name__ == '__main__':
    TestComparisonOperators.set_arguments()
    TestStringInformation.set_arguments()
    unittest.main()
