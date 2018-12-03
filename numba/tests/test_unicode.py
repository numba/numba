# -*- coding: utf-8 -*-

# This file tests Python 3.4 style unicode strings
# Tests should be skipped on Python < 3.4

from __future__ import print_function

import sys
from itertools import permutations

from numba import njit
import numba.unittest_support as unittest
from .support import (TestCase, no_pyobj_flags, MemoryLeakMixin)

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


if __name__ == '__main__':
    unittest.main()
