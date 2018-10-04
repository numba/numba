# This file tests Python 3.4 style unicode strings
# Tests should be skipped on Python < 3.4

from __future__ import print_function

import numba.unittest_support as unittest

import sys

from numba.compiler import compile_isolated, Flags
from numba import njit, types
import numba.unittest_support as unittest
from .support import (TestCase, no_pyobj_flags, MemoryLeakMixin)

# register everything
import numba.typing.unicode_str
import numba.targets.unicode_str


_py34_or_later = sys.version_info[:2] >= (3, 4)


def literal_usecase():
    return '大处着眼，小处着手。'


def passthrough_usecase(x):
    #print(x._length)
    #print(x._kind)
    #print('__')
    return x


def eq_usecase(x, y):
    return x == y


def len_usecase(x):
    return len(x)


def getitem_usecase(x, i):
    return x[i]


def slice2_usecase(x, i, j):
    return x[i:j]


def slice3_usecase(x, i, j, k):
    return x[i:j:k]


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
    'asc',
    'ascih',
    'ascii',
    'ascij',
    '大处着眼，小处着手',
    '大处着眼，小处着手。',
    '大处着眼，小处着手。🐍⚡',
]


# FIXME
UNICODE_EXAMPLES = [types.fake_str(x) for x in UNICODE_EXAMPLES]
UNICODE_ORDERING_EXAMPLES = [types.fake_str(x) for x in UNICODE_ORDERING_EXAMPLES]

@unittest.skipUnless(_py34_or_later, 'unicode support requires Python 3.4 or later')
class TestUnicode(BaseTest):

    #def test_literal(self, flags=no_pyobj_flags):
    #    pyfunc = literal_usecase
    #    self.run_nullary_func(pyfunc, flags=flags)

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
            self.assertEqual(pyfunc(a, a),
                             cfunc(a, a),
                             '%s: "%s", "%s"' % (usecase.__name__, a, a))

        # Check comparison to adjacent
        for a in UNICODE_ORDERING_EXAMPLES[:-1]:
            for b in UNICODE_ORDERING_EXAMPLES[1:]:
                self.assertEqual(pyfunc(a, b),
                                 cfunc(a, b),
                             '%s: "%s", "%s"' % (usecase.__name__, a, b))
                # and reversed
                self.assertEqual(pyfunc(b, a),
                                 cfunc(b, a),
                             '%s: "%s", "%s"' % (usecase.__name__, b, a))

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
            for b in [types.fake_str(x) for x in ['', 'x', a[:-2], a[3:], a, a + a]]:
                self.assertEqual(pyfunc(a, b),
                                 cfunc(a, b),
                                 '%s, %s' % (a, b))

    def test_endswith(self, flags=no_pyobj_flags):
        pyfunc = endswith_usecase
        cfunc = njit(pyfunc)
        for a in UNICODE_EXAMPLES:
            for b in [types.fake_str(x) for x in ['', 'x', a[:-2], a[3:], a, a + a]]:
                self.assertEqual(pyfunc(a, b),
                                 cfunc(a, b),
                                 '%s, %s' % (a, b))


    def test_in(self, flags=no_pyobj_flags):
        pyfunc = in_usecase
        cfunc = njit(pyfunc)
        for a in UNICODE_EXAMPLES:
            for substr in [types.fake_str(x) for x in ['', 'xx', a[::-1], a[:-2], a[3:], a, a + a]]:
                self.assertEqual(pyfunc(substr, a),
                                 cfunc(substr, a),
                                 "'%s' in '%s'?" % (substr, a))

    def test_find(self, flags=no_pyobj_flags):
        pyfunc = find_usecase
        cfunc = njit(pyfunc)
        for a in UNICODE_EXAMPLES:
            for substr in [types.fake_str(x) for x in ['', 'xx', a[::-1], a[:-2], a[3:], a, a + a]]:
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

    def test_slice2(self):
        pyfunc = slice2_usecase
        cfunc = njit(pyfunc)

        for s in UNICODE_EXAMPLES:
            for i in range(-len(s), len(s)):
                for j in range(-len(s), len(s)):
                    self.assertEqual(pyfunc(s, i, j),
                                    cfunc(s, i, j),
                                    "'%s'[%d:%d]?" % (s, i, j))

    def test_slice3(self):
        pyfunc = slice3_usecase
        cfunc = njit(pyfunc)

        for s in UNICODE_EXAMPLES:
            for i in range(-len(s), len(s)):
                for j in range(-len(s), len(s)):
                    for k in [-2, -1, 1, 2]:
                        self.assertEqual(pyfunc(s, i, j, k),
                                        cfunc(s, i, j, k),
                                        "'%s'[%d:%d:%d]?" % (s, i, j, k))

    def test_concat(self, flags=no_pyobj_flags):
        pyfunc = concat_usecase
        cfunc = njit(pyfunc)
        for a in UNICODE_EXAMPLES:
            for b in UNICODE_EXAMPLES[::-1]:
                self.assertEqual(pyfunc(a, b),
                                 cfunc(a, b),
                                 "'%s' + '%s'?" % (a, b))



if __name__ == '__main__':
    unittest.main()
