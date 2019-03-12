# -*- coding: utf-8 -*-

# This file tests Python 3.4 style unicode strings
# Tests should be skipped on Python < 3.4

from __future__ import print_function

import sys
from itertools import permutations

from numba import njit
import numba.unittest_support as unittest
from .support import (TestCase, no_pyobj_flags, MemoryLeakMixin)
from numba.errors import TypingError

_py34_or_later = sys.version_info[:2] >= (3, 4)


def literal_usecase():
    return '大处着眼，小处着手。'


def passthrough_usecase(x):
    return x


def eq_usecase(x, y):
    return x == y


def len_usecase(x):
    return len(x)


def getitem_usecase(x, i):
    return x[i]


def concat_usecase(x, y):
    return x + y


def inplace_concat_usecase(x, y):
    x += y
    return x


def in_usecase(x, y):
    return x in y


def lt_usecase(x, y):
    return x < y


def le_usecase(x, y):
    return x <= y


def gt_usecase(x, y):
    return x > y


def ge_usecase(x, y):
    return x >= y


def find_usecase(x, y):
    return x.find(y)


def startswith_usecase(x, y):
    return x.startswith(y)


def endswith_usecase(x, y):
    return x.endswith(y)


def split_usecase(x, y):
    return x.split(y)


def split_with_maxsplit_usecase(x, y, maxsplit):
    return x.split(y, maxsplit)


def split_with_maxsplit_kwarg_usecase(x, y, maxsplit):
    return x.split(y, maxsplit=maxsplit)


def split_whitespace_usecase(x):
    return x.split()


def join_usecase(x, y):
    return x.join(y)


def join_empty_usecase(x):
    # hack to make empty typed list
    l = ['']
    l.pop()
    return x.join(l)


class BaseTest(MemoryLeakMixin, TestCase):
    def setUp(self):
        super(BaseTest, self).setUp()


UNICODE_EXAMPLES = [
    'ascii',
    '12345',
    '1234567890',
    '¡Y tú quién te crees?',
    '🐍⚡',
    '大处着眼，小处着手。',
]

UNICODE_ORDERING_EXAMPLES = [
    '',
    'a'
    'aa',
    'aaa',
    'b',
    'aab',
    'ab',
    'asc',
    'ascih',
    'ascii',
    'ascij',
    '大处着眼，小处着手',
    '大处着眼，小处着手。',
    '大处着眼，小处着手。🐍⚡',
]


@unittest.skipUnless(_py34_or_later,
                     'unicode support requires Python 3.4 or later')
class TestUnicode(BaseTest):

    def test_literal(self, flags=no_pyobj_flags):
        pyfunc = literal_usecase
        self.run_nullary_func(pyfunc, flags=flags)

    def test_passthrough(self, flags=no_pyobj_flags):
        pyfunc = passthrough_usecase
        cfunc = njit(pyfunc)
        for s in UNICODE_EXAMPLES:
            self.assertEqual(pyfunc(s), cfunc(s))

    def test_eq(self, flags=no_pyobj_flags):
        pyfunc = eq_usecase
        cfunc = njit(pyfunc)
        for a in UNICODE_EXAMPLES:
            for b in reversed(UNICODE_EXAMPLES):
                self.assertEqual(pyfunc(a, b),
                                 cfunc(a, b), '%s, %s' % (a, b))

    def _check_ordering_op(self, usecase):
        pyfunc = usecase
        cfunc = njit(pyfunc)

        # Check comparison to self
        for a in UNICODE_ORDERING_EXAMPLES:
            self.assertEqual(
                pyfunc(a, a),
                cfunc(a, a),
                '%s: "%s", "%s"' % (usecase.__name__, a, a),
            )

        # Check comparison to adjacent
        for a, b in permutations(UNICODE_ORDERING_EXAMPLES, r=2):
            self.assertEqual(
                pyfunc(a, b),
                cfunc(a, b),
                '%s: "%s", "%s"' % (usecase.__name__, a, b),
            )
            # and reversed
            self.assertEqual(
                pyfunc(b, a),
                cfunc(b, a),
                '%s: "%s", "%s"' % (usecase.__name__, b, a),
            )

    def test_lt(self, flags=no_pyobj_flags):
        self._check_ordering_op(lt_usecase)

    def test_le(self, flags=no_pyobj_flags):
        self._check_ordering_op(le_usecase)

    def test_gt(self, flags=no_pyobj_flags):
        self._check_ordering_op(gt_usecase)

    def test_ge(self, flags=no_pyobj_flags):
        self._check_ordering_op(ge_usecase)

    def test_len(self, flags=no_pyobj_flags):
        pyfunc = len_usecase
        cfunc = njit(pyfunc)
        for s in UNICODE_EXAMPLES:
            self.assertEqual(pyfunc(s), cfunc(s))

    def test_startswith(self, flags=no_pyobj_flags):
        pyfunc = startswith_usecase
        cfunc = njit(pyfunc)
        for a in UNICODE_EXAMPLES:
            for b in [x for x in ['', 'x', a[:-2], a[3:], a, a + a]]:
                self.assertEqual(pyfunc(a, b),
                                 cfunc(a, b),
                                 '%s, %s' % (a, b))

    def test_endswith(self, flags=no_pyobj_flags):
        pyfunc = endswith_usecase
        cfunc = njit(pyfunc)
        for a in UNICODE_EXAMPLES:
            for b in [x for x in ['', 'x', a[:-2], a[3:], a, a + a]]:
                self.assertEqual(pyfunc(a, b),
                                 cfunc(a, b),
                                 '%s, %s' % (a, b))

    def test_in(self, flags=no_pyobj_flags):
        pyfunc = in_usecase
        cfunc = njit(pyfunc)
        for a in UNICODE_EXAMPLES:
            extras = ['', 'xx', a[::-1], a[:-2], a[3:], a, a + a]
            for substr in [x for x in extras]:
                self.assertEqual(pyfunc(substr, a),
                                 cfunc(substr, a),
                                 "'%s' in '%s'?" % (substr, a))

    def test_find(self, flags=no_pyobj_flags):
        pyfunc = find_usecase
        cfunc = njit(pyfunc)
        for a in UNICODE_EXAMPLES:
            extras = ['', 'xx', a[::-1], a[:-2], a[3:], a, a + a]
            for substr in [x for x in extras]:
                self.assertEqual(pyfunc(a, substr),
                                 cfunc(a, substr),
                                 "'%s'.find('%s')?" % (a, substr))

    def test_getitem(self):
        pyfunc = getitem_usecase
        cfunc = njit(pyfunc)

        for s in UNICODE_EXAMPLES:
            for i in range(-len(s)):
                self.assertEqual(pyfunc(s, i),
                                 cfunc(s, i),
                                 "'%s'[%d]?" % (s, i))

    def test_getitem_error(self):
        self.disable_leak_check()

        pyfunc = getitem_usecase
        cfunc = njit(pyfunc)

        for s in UNICODE_EXAMPLES:
            with self.assertRaises(IndexError) as raises:
                pyfunc(s, len(s))
            self.assertIn('string index out of range', str(raises.exception))

            with self.assertRaises(IndexError) as raises:
                cfunc(s, len(s))
            self.assertIn('string index out of range', str(raises.exception))

    def test_slice2(self):
        pyfunc = getitem_usecase
        cfunc = njit(pyfunc)

        for s in UNICODE_EXAMPLES:
            for i in list(range(-len(s), len(s))):
                for j in list(range(-len(s), len(s))):
                    sl = slice(i, j)
                    self.assertEqual(pyfunc(s, sl),
                                     cfunc(s, sl),
                                     "'%s'[%d:%d]?" % (s, i, j))

    def test_slice2_error(self):
        pyfunc = getitem_usecase
        cfunc = njit(pyfunc)

        for s in UNICODE_EXAMPLES:
            for i in [-2, -1, len(s), len(s) + 1]:
                for j in [-2, -1, len(s), len(s) + 1]:
                    sl = slice(i, j)
                    self.assertEqual(pyfunc(s, sl),
                                     cfunc(s, sl),
                                     "'%s'[%d:%d]?" % (s, i, j))

    def test_slice3(self):
        pyfunc = getitem_usecase
        cfunc = njit(pyfunc)

        for s in UNICODE_EXAMPLES:
            for i in range(-len(s), len(s)):
                for j in range(-len(s), len(s)):
                    for k in [-2, -1, 1, 2]:
                        sl = slice(i, j, k)
                        self.assertEqual(pyfunc(s, sl),
                                         cfunc(s, sl),
                                         "'%s'[%d:%d:%d]?" % (s, i, j, k))

    def test_slice3_error(self):
        pyfunc = getitem_usecase
        cfunc = njit(pyfunc)

        for s in UNICODE_EXAMPLES:
            for i in [-2, -1, len(s), len(s) + 1]:
                for j in [-2, -1, len(s), len(s) + 1]:
                    for k in [-2, -1, 1, 2]:
                        sl = slice(i, j, k)
                        self.assertEqual(pyfunc(s, sl),
                                         cfunc(s, sl),
                                         "'%s'[%d:%d:%d]?" % (s, i, j, k))

    def test_concat(self, flags=no_pyobj_flags):
        pyfunc = concat_usecase
        cfunc = njit(pyfunc)
        for a in UNICODE_EXAMPLES:
            for b in UNICODE_EXAMPLES[::-1]:
                self.assertEqual(pyfunc(a, b),
                                 cfunc(a, b),
                                 "'%s' + '%s'?" % (a, b))

    def test_split_exception_empty_sep(self):
        self.disable_leak_check()

        pyfunc = split_usecase
        cfunc = njit(pyfunc)

        # Handle empty separator exception
        for func in [pyfunc, cfunc]:
            with self.assertRaises(ValueError) as raises:
                func('a', '')
            self.assertIn('empty separator', str(raises.exception))

    def test_split_exception_noninteger_maxsplit(self):
        pyfunc = split_with_maxsplit_usecase
        cfunc = njit(pyfunc)

        # Handle non-integer maxsplit exception
        for sep in [' ', None]:
            with self.assertRaises(TypingError) as raises:
                cfunc('a', sep, 2.4)
            self.assertIn('float64', str(raises.exception),
                          'non-integer maxsplit with sep = %s' % sep)

    def test_split(self):
        pyfunc = split_usecase
        cfunc = njit(pyfunc)

        CASES = [
            (' a ', None),
            ('', '⚡'),
            ('abcabc', '⚡'),
            ('🐍⚡', '⚡'),
            ('🐍⚡🐍', '⚡'),
            ('abababa', 'a'),
            ('abababa', 'b'),
            ('abababa', 'c'),
            ('abababa', 'ab'),
            ('abababa', 'aba'),
        ]

        for test_str, splitter in CASES:
            self.assertEqual(pyfunc(test_str, splitter),
                             cfunc(test_str, splitter),
                             "'%s'.split('%s')?" % (test_str, splitter))

    def test_split_with_maxsplit(self):
        CASES = [
            (' a ', None, 1),
            ('', '⚡', 1),
            ('abcabc', '⚡', 1),
            ('🐍⚡', '⚡', 1),
            ('🐍⚡🐍', '⚡', 1),
            ('abababa', 'a', 2),
            ('abababa', 'b', 1),
            ('abababa', 'c', 2),
            ('abababa', 'ab', 1),
            ('abababa', 'aba', 5),
        ]

        for pyfunc, fmt_str in [(split_with_maxsplit_usecase, "'%s'.split('%s', %d)?"),
                                (split_with_maxsplit_kwarg_usecase, "'%s'.split('%s', maxsplit=%d)?")]:

            cfunc = njit(pyfunc)
            for test_str, splitter, maxsplit in CASES:
                self.assertEqual(pyfunc(test_str, splitter, maxsplit),
                                 cfunc(test_str, splitter, maxsplit),
                                 fmt_str % (test_str, splitter, maxsplit))

    def test_split_whitespace(self):
        # explicit sep=None cases covered in test_split and test_split_with_maxsplit
        pyfunc = split_whitespace_usecase
        cfunc = njit(pyfunc)

        #list copied from https://github.com/python/cpython/blob/master/Objects/unicodetype_db.h
        all_whitespace = ''.join(map(chr, [
            0x0009, 0x000A, 0x000B, 0x000C, 0x000D, 0x001C, 0x001D, 0x001E, 0x001F, 0x0020,
            0x0085, 0x00A0, 0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006,
            0x2007, 0x2008, 0x2009, 0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000
        ]))

        CASES = [
            '',
            'abcabc',
            '🐍 ⚡',
            '🐍 ⚡ 🐍',
            '🐍   ⚡ 🐍  ',
            '  🐍   ⚡ 🐍',
            ' 🐍' + all_whitespace + '⚡ 🐍  ',
        ]
        for test_str in CASES:
            self.assertEqual(pyfunc(test_str),
                             cfunc(test_str),
                             "'%s'.split()?" % (test_str,))

    def test_join_empty(self):
        # Can't pass empty list to nopython mode, so we have to make a
        # separate test case
        pyfunc = join_empty_usecase
        cfunc = njit(pyfunc)

        CASES = [
            '',
            '🐍🐍🐍',
        ]

        for sep in CASES:
            self.assertEqual(pyfunc(sep),
                             cfunc(sep),
                             "'%s'.join([])?" % (sep,))

    def test_join_non_string_exception(self):
        # Verify that join of list of integers raises typing exception
        pyfunc = join_usecase
        cfunc = njit(pyfunc)

        # Handle empty separator exception
        with self.assertRaises(TypingError) as raises:
            cfunc('', [1,2,3])
        # This error message is obscure, but indicates the error was trapped in typing of str.join()
        # Feel free to change this as we update error messages.
        exc_message = str(raises.exception)
        self.assertIn("Invalid use of BoundFunction", exc_message)
        self.assertIn("(reflected list(int", exc_message)  # could be int32 or int64

    def test_join(self):
        pyfunc = join_usecase
        cfunc = njit(pyfunc)

        CASES = [
            ('', ['', '', '']),
            ('a', ['', '', '']),
            ('', ['a', 'bbbb', 'c']),
            ('🐍🐍🐍', ['⚡⚡'] * 5),
        ]

        for sep, parts in CASES:
            self.assertEqual(pyfunc(sep, parts),
                             cfunc(sep, parts),
                             "'%s'.join('%s')?" % (sep, parts))

    def test_join_interleave_str(self):
        # can pass a string as the parts iterable
        pyfunc = join_usecase
        cfunc = njit(pyfunc)

        CASES = [
            ('abc', '123'),
            ('🐍🐍🐍', '⚡⚡'),
        ]

        for sep, parts in CASES:
            self.assertEqual(pyfunc(sep, parts),
                             cfunc(sep, parts),
                             "'%s'.join('%s')?" % (sep, parts))

    def test_inplace_concat(self, flags=no_pyobj_flags):
        pyfunc = inplace_concat_usecase
        cfunc = njit(pyfunc)
        for a in UNICODE_EXAMPLES:
            for b in UNICODE_EXAMPLES[::-1]:
                self.assertEqual(pyfunc(a, b),
                                 cfunc(a, b),
                                 "'%s' + '%s'?" % (a, b))

    def test_pointless_slice(self, flags=no_pyobj_flags):
        def pyfunc(a):
            return a[:]
        cfunc = njit(pyfunc)
        args = ['a']
        self.assertEqual(pyfunc(*args), cfunc(*args))

    def test_walk_backwards(self, flags=no_pyobj_flags):
        def pyfunc(a):
            return a[::-1]
        cfunc = njit(pyfunc)
        args = ['a']
        self.assertEqual(pyfunc(*args), cfunc(*args))

    def test_stride_slice(self, flags=no_pyobj_flags):
        def pyfunc(a):
            return a[::2]
        cfunc = njit(pyfunc)
        args = ['a']
        self.assertEqual(pyfunc(*args), cfunc(*args))

    def test_basic_lt(self, flags=no_pyobj_flags):
        def pyfunc(a, b):
            return a < b
        cfunc = njit(pyfunc)
        args = ['ab', 'b']
        self.assertEqual(pyfunc(*args), cfunc(*args))

    def test_basic_gt(self, flags=no_pyobj_flags):
        def pyfunc(a, b):
            return a > b
        cfunc = njit(pyfunc)
        args = ['ab', 'b']
        self.assertEqual(pyfunc(*args), cfunc(*args))

    def test_comparison(self):
        def pyfunc(option, x, y):
            if option == '==':
                return x == y
            elif option == '!=':
                return x != y
            elif option == '<':
                return x < y
            elif option == '>':
                return x > y
            elif option == '<=':
                return x <= y
            elif option == '>=':
                return x >= y
            else:
                return None

        cfunc = njit(pyfunc)

        for x, y in permutations(UNICODE_ORDERING_EXAMPLES, r=2):
            for cmpop in ['==', '!=', '<', '>', '<=', '>=', '']:
                args = [cmpop, x, y]
                self.assertEqual(pyfunc(*args), cfunc(*args),
                                 msg='failed on {}'.format(args))

    def test_literal_concat(self):
        def pyfunc(x):
            abc = 'abc'
            if len(x):
                return abc + 'b123' + x + 'IO'
            else:
                return x + abc + '123' + x

        cfunc = njit(pyfunc)
        args = ['x']
        self.assertEqual(pyfunc(*args), cfunc(*args))
        args = ['']
        self.assertEqual(pyfunc(*args), cfunc(*args))

    def test_literal_comparison(self):
        def pyfunc(option):
            x = 'a123'
            y = 'aa12'
            if option == '==':
                return x == y
            elif option == '!=':
                return x != y
            elif option == '<':
                return x < y
            elif option == '>':
                return x > y
            elif option == '<=':
                return x <= y
            elif option == '>=':
                return x >= y
            else:
                return None

        cfunc = njit(pyfunc)
        for cmpop in ['==', '!=', '<', '>', '<=', '>=', '']:
            args = [cmpop]
            self.assertEqual(pyfunc(*args), cfunc(*args),
                             msg='failed on {}'.format(args))

    def test_literal_len(self):
        def pyfunc():
            return len('abc')
        cfunc = njit(pyfunc)
        self.assertEqual(pyfunc(), cfunc())

    def test_literal_getitem(self):
        def pyfunc(which):
            return 'abc'[which]
        cfunc = njit(pyfunc)
        for a in [-1, 0, 1, slice(1, None), slice(None, -1)]:
            args = [a]
            self.assertEqual(pyfunc(*args), cfunc(*args),
                             msg='failed on {}'.format(args))

    def test_literal_in(self):
        def pyfunc(x):
            return x in '9876zabiuh'

        cfunc = njit(pyfunc)
        for a in ['a', '9', '1', '', '8uha', '987']:
            args = [a]
            self.assertEqual(pyfunc(*args), cfunc(*args),
                             msg='failed on {}'.format(args))

    def test_literal_xyzwith(self):
        def pyfunc(x, y):
            return 'abc'.startswith(x), 'cde'.endswith(y)

        cfunc = njit(pyfunc)
        for args in permutations('abcdefg', r=2):
            self.assertEqual(pyfunc(*args), cfunc(*args),
                             msg='failed on {}'.format(args))

    def test_literal_find(self):
        def pyfunc(x):
            return 'abc'.find(x), x.find('a')

        cfunc = njit(pyfunc)
        for a in ['ab']:
            args = [a]
            self.assertEqual(pyfunc(*args), cfunc(*args),
                             msg='failed on {}'.format(args))


@unittest.skipUnless(_py34_or_later,
                     'unicode support requires Python 3.4 or later')
class TestUnicodeInTuple(BaseTest):

    def test_const_unicode_in_tuple(self):
        # Issue 3673
        @njit
        def f():
            return ('aa',) < ('bb',)

        self.assertEqual(f.py_func(), f())

        @njit
        def f():
            return ('cc',) < ('bb',)

        self.assertEqual(f.py_func(), f())

    def test_const_unicode_in_hetero_tuple(self):
        @njit
        def f():
            return ('aa', 1) < ('bb', 1)

        self.assertEqual(f.py_func(), f())

        @njit
        def f():
            return ('aa', 1) < ('aa', 2)

        self.assertEqual(f.py_func(), f())


if __name__ == '__main__':
    unittest.main()
