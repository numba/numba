# -*- coding: utf-8 -*-
import sys
import unittest
from itertools import permutations
from itertools import product

from numba import njit
from numba.core import types
from numba.core.errors import TypingError, UnsupportedError
from numba.core.types.functions import _header_lead
from numba.cpython.unicode import _MAX_UNICODE, _INVALID_LITERAL_MSG
from numba.extending import overload
from numba.tests.support import (TestCase, no_pyobj_flags, MemoryLeakMixin)


def isascii(s):
    return all(ord(c) < 128 for c in s)


def literal_usecase():
    return '大处着眼，小处着手。'


def passthrough_usecase(x):
    return x


def eq_usecase(x, y):
    return x == y


def len_usecase(x):
    return len(x)


def bool_usecase(x):
    return bool(x)


def getitem_usecase(x, i):
    return x[i]


def getitem_check_kind_usecase(x, i):
    return hash(x[i])


def zfill_usecase(x, y):
    return x.zfill(y)


def concat_usecase(x, y):
    return x + y


def repeat_usecase(x, y):
    return x * y


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


def partition_usecase(s, sep):
    return s.partition(sep)


def find_usecase(x, y):
    return x.find(y)


def find_with_start_only_usecase(x, y, start):
    return x.find(y, start)


def find_with_start_end_usecase(x, y, start, end):
    return x.find(y, start, end)


def rpartition_usecase(s, sep):
    return s.rpartition(sep)


def count_usecase(x, y):
    return x.count(y)


def count_with_start_usecase(x, y, start):
    return x.count(y, start)


def count_with_start_end_usecase(x, y, start, end):
    return x.count(y, start, end)


def rfind_usecase(x, y):
    return x.rfind(y)


def rfind_with_start_only_usecase(x, y, start):
    return x.rfind(y, start)


def rfind_with_start_end_usecase(x, y, start, end):
    return x.rfind(y, start, end)


def replace_usecase(s, x, y):
    return s.replace(x, y)


def replace_with_count_usecase(s, x, y, count):
    return s.replace(x, y, count)


def rindex_usecase(x, y):
    return x.rindex(y)


def rindex_with_start_only_usecase(x, y, start):
    return x.rindex(y, start)


def rindex_with_start_end_usecase(x, y, start, end):
    return x.rindex(y, start, end)


def index_usecase(x, y):
    return x.index(y)


def index_with_start_only_usecase(x, y, start):
    return x.index(y, start)


def index_with_start_end_usecase(x, y, start, end):
    return x.index(y, start, end)


def startswith_usecase(x, y):
    return x.startswith(y)


def endswith_usecase(x, y):
    return x.endswith(y)


def expandtabs_usecase(s):
    return s.expandtabs()


def expandtabs_with_tabsize_usecase(s, tabsize):
    return s.expandtabs(tabsize)


def expandtabs_with_tabsize_kwarg_usecase(s, tabsize):
    return s.expandtabs(tabsize=tabsize)


def startswith_with_start_only_usecase(x, y, start):
    return x.startswith(y, start)


def startswith_with_start_end_usecase(x, y, start, end):
    return x.startswith(y, start, end)


def endswith_with_start_only_usecase(x, y, start):
    return x.endswith(y, start)


def endswith_with_start_end_usecase(x, y, start, end):
    return x.endswith(y, start, end)


def split_usecase(x, y):
    return x.split(y)


def split_with_maxsplit_usecase(x, y, maxsplit):
    return x.split(y, maxsplit)


def split_with_maxsplit_kwarg_usecase(x, y, maxsplit):
    return x.split(y, maxsplit=maxsplit)


def split_whitespace_usecase(x):
    return x.split()


def splitlines_usecase(s):
    return s.splitlines()


def splitlines_with_keepends_usecase(s, keepends):
    return s.splitlines(keepends)


def splitlines_with_keepends_kwarg_usecase(s, keepends):
    return s.splitlines(keepends=keepends)


def rsplit_usecase(s, sep):
    return s.rsplit(sep)


def rsplit_with_maxsplit_usecase(s, sep, maxsplit):
    return s.rsplit(sep, maxsplit)


def rsplit_with_maxsplit_kwarg_usecase(s, sep, maxsplit):
    return s.rsplit(sep, maxsplit=maxsplit)


def rsplit_whitespace_usecase(s):
    return s.rsplit()


def lstrip_usecase(x):
    return x.lstrip()


def lstrip_usecase_chars(x, chars):
    return x.lstrip(chars)


def rstrip_usecase(x):
    return x.rstrip()


def rstrip_usecase_chars(x, chars):
    return x.rstrip(chars)


def strip_usecase(x):
    return x.strip()


def strip_usecase_chars(x, chars):
    return x.strip(chars)


def join_usecase(x, y):
    return x.join(y)


def join_empty_usecase(x):
    # hack to make empty typed list
    l = ['']
    l.pop()
    return x.join(l)


def center_usecase(x, y):
    return x.center(y)


def center_usecase_fillchar(x, y, fillchar):
    return x.center(y, fillchar)


def ljust_usecase(x, y):
    return x.ljust(y)


def ljust_usecase_fillchar(x, y, fillchar):
    return x.ljust(y, fillchar)


def rjust_usecase(x, y):
    return x.rjust(y)


def rjust_usecase_fillchar(x, y, fillchar):
    return x.rjust(y, fillchar)


def istitle_usecase(x):
    return x.istitle()


def iter_usecase(x):
    l = []
    for i in x:
        l.append(i)
    return l


def title(x):
    return x.title()


def literal_iter_usecase():
    l = []
    for i in '大处着眼，小处着手。':
        l.append(i)
    return l


def enumerated_iter_usecase(x):
    buf = ""
    scan = 0
    for i, s in enumerate(x):
        buf += s
        scan += 1
    return buf, scan


def iter_stopiteration_usecase(x):
    n = len(x)
    i = iter(x)
    for _ in range(n + 1):
        next(i)


def literal_iter_stopiteration_usecase():
    s = '大处着眼，小处着手。'
    i = iter(s)
    n = len(s)
    for _ in range(n + 1):
        next(i)


def islower_usecase(x):
    return x.islower()


def lower_usecase(x):
    return x.lower()


def ord_usecase(x):
    return ord(x)


def chr_usecase(x):
    return chr(x)


def int_usecase(s):
    return int(s)


def int_base_usecase(s, base):
    return int(s, base)


class BaseTest(MemoryLeakMixin, TestCase):
    def setUp(self):
        super(BaseTest, self).setUp()


UNICODE_EXAMPLES = [
    '',
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

UNICODE_COUNT_EXAMPLES = [
    ('', ''),
    ('', 'ascii'),
    ('ascii', ''),
    ('asc ii', ' '),
    ('ascii', 'ci'),
    ('ascii', 'ascii'),
    ('ascii', 'Ă'),
    ('ascii', '大处'),
    ('ascii', 'étú?'),
    ('', '大处 着眼，小处着手。大大大处'),
    ('大处 着眼，小处着手。大大大处', ''),
    ('大处 着眼，小处着手。大大大处', ' '),
    ('大处 着眼，小处着手。大大大处', 'ci'),
    ('大处 着眼，小处着手。大大大处', '大处大处'),
    ('大处 着眼，小处着手。大大大处', '大处 着眼，小处着手。大大大处'),
    ('大处 着眼，小处着手。大大大处', 'Ă'),
    ('大处 着眼，小处着手。大大大处', '大处'),
    ('大处 着眼，小处着手。大大大处', 'étú?'),
    ('', 'tú quién te crees?'),
    ('tú quién te crees?', ''),
    ('tú quién te crees?', ' '),
    ('tú quién te crees?', 'ci'),
    ('tú quién te crees?', 'tú quién te crees?'),
    ('tú quién te crees?', 'Ă'),
    ('tú quién te crees?', '大处'),
    ('tú quién te crees?', 'étú?'),
    ('abababab', 'a'),
    ('abababab', 'ab'),
    ('abababab', 'aba'),
    ('aaaaaaaaaa', 'aaa'),
    ('aaaaaaaaaa', 'aĂ'),
    ('aabbaaaabbaa', 'aa')
]


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
                # comparing against something that's not unicode
                self.assertEqual(pyfunc(a, 1),
                                 cfunc(a, 1), '%s, %s' % (a, 1))
                self.assertEqual(pyfunc(1, b),
                                 cfunc(1, b), '%s, %s' % (1, b))

    def test_eq_optional(self):
        # See issue #7474
        @njit
        def foo(pred1, pred2):
            if pred1 > 0:
                resolved1 = 'concrete'
            else:
                resolved1 = None
            if pred2 < 0:
                resolved2 = 'concrete'
            else:
                resolved2 = None

            # resolved* are Optionals
            if resolved1 == resolved2:
                return 10
            else:
                return 20

        for (p1, p2) in product(*((-1, 1),) * 2):
            self.assertEqual(foo(p1, p2), foo.py_func(p1, p2))

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

    def test_bool(self, flags=no_pyobj_flags):
        pyfunc = bool_usecase
        cfunc = njit(pyfunc)
        for s in UNICODE_EXAMPLES:
            self.assertEqual(pyfunc(s), cfunc(s))

    def test_expandtabs(self):
        pyfunc = expandtabs_usecase
        cfunc = njit(pyfunc)

        cases = ['', '\t', 't\tt\t', 'a\t', '\t⚡', 'a\tbc\nab\tc',
                 '🐍\t⚡', '🐍⚡\n\t\t🐍\t', 'ab\rab\t\t\tab\r\n\ta']

        msg = 'Results of "{}".expandtabs() must be equal'
        for s in cases:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

    def test_expandtabs_with_tabsize(self):
        fns = [njit(expandtabs_with_tabsize_usecase),
               njit(expandtabs_with_tabsize_kwarg_usecase)]
        messages = ['Results of "{}".expandtabs({}) must be equal',
                    'Results of "{}".expandtabs(tabsize={}) must be equal']

        cases = ['', '\t', 't\tt\t', 'a\t', '\t⚡', 'a\tbc\nab\tc',
                 '🐍\t⚡', '🐍⚡\n\t\t🐍\t', 'ab\rab\t\t\tab\r\n\ta']

        for s in cases:
            for tabsize in range(-1, 10):
                for fn, msg in zip(fns, messages):
                    self.assertEqual(fn.py_func(s, tabsize), fn(s, tabsize),
                                     msg=msg.format(s, tabsize))

    def test_expandtabs_exception_noninteger_tabsize(self):
        pyfunc = expandtabs_with_tabsize_usecase
        cfunc = njit(pyfunc)

        accepted_types = (types.Integer, int)
        with self.assertRaises(TypingError) as raises:
            cfunc('\t', 2.4)
        msg = '"tabsize" must be {}, not float'.format(accepted_types)
        self.assertIn(msg, str(raises.exception))

    def test_startswith_default(self):
        pyfunc = startswith_usecase
        cfunc = njit(pyfunc)

        cpython_str = ['hello', 'helloworld', '']
        cpython_subs = [
            'he', 'hello', 'helloworld', 'ello',
            '', 'lowo', 'lo', 'he', 'lo', 'o',
        ]
        extra_subs = ['hellohellohello', ' ']
        for s in cpython_str + UNICODE_EXAMPLES:
            default_subs = ['', 'x', s[:-2], s[3:], s, s + s]
            for prefix in cpython_subs + default_subs + extra_subs:
                self.assertEqual(pyfunc(s, prefix), cfunc(s, prefix))

    def test_startswith_with_start(self):
        pyfunc = startswith_with_start_only_usecase
        cfunc = njit(pyfunc)

        cpython_str = ['hello', 'helloworld', '']
        cpython_subs = [
            'he', 'hello', 'helloworld', 'ello',
            '', 'lowo', 'lo', 'he', 'lo', 'o',
        ]
        extra_subs = ['hellohellohello', ' ']
        for s in cpython_str + UNICODE_EXAMPLES:
            default_subs = ['', 'x', s[:-2], s[3:], s, s + s]
            for prefix in cpython_subs + default_subs + extra_subs:
                for start in list(range(-20, 20)) + [None]:
                    self.assertEqual(pyfunc(s, prefix, start),
                                     cfunc(s, prefix, start))

    def test_startswith_with_start_end(self):
        pyfunc = startswith_with_start_end_usecase
        cfunc = njit(pyfunc)

        cpython_str = ['hello', 'helloworld', '']
        cpython_subs = [
            'he', 'hello', 'helloworld', 'ello',
            '', 'lowo', 'lo', 'he', 'lo', 'o',
        ]
        extra_subs = ['hellohellohello', ' ']
        for s in cpython_str + UNICODE_EXAMPLES:
            default_subs = ['', 'x', s[:-2], s[3:], s, s + s]
            for prefix in cpython_subs + default_subs + extra_subs:
                for start in list(range(-20, 20)) + [None]:
                    for end in list(range(-20, 20)) + [None]:
                        self.assertEqual(pyfunc(s, prefix, start, end),
                                         cfunc(s, prefix, start, end))

    def test_startswith_exception_invalid_args(self):
        msg_invalid_prefix = \
            "The arg 'prefix' should be a string or a tuple of strings"
        with self.assertRaisesRegex(TypingError, msg_invalid_prefix):
            cfunc = njit(startswith_usecase)
            cfunc("hello", (1, "he"))

        msg_invalid_start = \
            "When specified, the arg 'start' must be an Integer or None"
        with self.assertRaisesRegex(TypingError, msg_invalid_start):
            cfunc = njit(startswith_with_start_only_usecase)
            cfunc("hello", "he", "invalid start")

        msg_invalid_end = \
            "When specified, the arg 'end' must be an Integer or None"
        with self.assertRaisesRegex(TypingError, msg_invalid_end):
            cfunc = njit(startswith_with_start_end_usecase)
            cfunc("hello", "he", 0, "invalid end")

    def test_startswith_tuple(self):
        pyfunc = startswith_usecase
        cfunc = njit(pyfunc)

        cpython_str = ['hello', 'helloworld', '']
        cpython_subs = [
            'he', 'hello', 'helloworld', 'ello',
            '', 'lowo', 'lo', 'he', 'lo', 'o',
        ]
        extra_subs = ['hellohellohello', ' ']
        for s in cpython_str + UNICODE_EXAMPLES:
            default_subs = ['', 'x', s[:-2], s[3:], s, s + s]
            for sub_str in cpython_subs + default_subs + extra_subs:
                prefix = (sub_str, 'lo')
                self.assertEqual(pyfunc(s, prefix),
                                 cfunc(s, prefix))

    def test_startswith_tuple_args(self):
        pyfunc = startswith_with_start_end_usecase
        cfunc = njit(pyfunc)

        cpython_str = ['hello', 'helloworld', '']
        cpython_subs = [
            'he', 'hello', 'helloworld', 'ello',
            '', 'lowo', 'lo', 'he', 'lo', 'o',
        ]
        extra_subs = ['hellohellohello', ' ']
        for s in cpython_str + UNICODE_EXAMPLES:
            default_subs = ['', 'x', s[:-2], s[3:], s, s + s]
            for sub_str in cpython_subs + default_subs + extra_subs:
                for start in list(range(-20, 20)) + [None]:
                    for end in list(range(-20, 20)) + [None]:
                        prefix = (sub_str, 'lo')
                        self.assertEqual(pyfunc(s, prefix, start, end),
                                         cfunc(s, prefix, start, end))

    def test_endswith_default(self):
        pyfunc = endswith_usecase
        cfunc = njit(pyfunc)

        # Samples taken from CPython testing:
        # https://github.com/python/cpython/blob/865c3b257fe38154a4320c7ee6afb416f665b9c2/Lib/test/string_tests.py#L1049-L1099    # noqa: E501
        cpython_str = ['hello', 'helloworld', '']
        cpython_subs = [
            'he', 'hello', 'helloworld', 'ello',
            '', 'lowo', 'lo', 'he', 'lo', 'o',
        ]
        extra_subs = ['hellohellohello', ' ']
        for s in cpython_str + UNICODE_EXAMPLES:
            default_subs = ['', 'x', s[:-2], s[3:], s, s + s]
            for sub_str in cpython_subs + default_subs + extra_subs:
                msg = 'Results "{}".endswith("{}") must be equal'
                self.assertEqual(pyfunc(s, sub_str), cfunc(s, sub_str),
                                 msg=msg.format(s, sub_str))

    def test_endswith_with_start(self):
        pyfunc = endswith_with_start_only_usecase
        cfunc = njit(pyfunc)

        # Samples taken from CPython testing:
        # https://github.com/python/cpython/blob/865c3b257fe38154a4320c7ee6afb416f665b9c2/Lib/test/string_tests.py#L1049-L1099    # noqa: E501
        cpython_str = ['hello', 'helloworld', '']
        cpython_subs = [
            'he', 'hello', 'helloworld', 'ello',
            '', 'lowo', 'lo', 'he', 'lo', 'o',
        ]
        extra_subs = ['hellohellohello', ' ']
        for s in cpython_str + UNICODE_EXAMPLES:
            default_subs = ['', 'x', s[:-2], s[3:], s, s + s]
            for sub_str in cpython_subs + default_subs + extra_subs:
                for start in list(range(-20, 20)) + [None]:
                    msg = 'Results "{}".endswith("{}", {}) must be equal'
                    self.assertEqual(pyfunc(s, sub_str, start),
                                     cfunc(s, sub_str, start),
                                     msg=msg.format(s, sub_str, start))

    def test_endswith_with_start_end(self):
        pyfunc = endswith_with_start_end_usecase
        cfunc = njit(pyfunc)

        # Samples taken from CPython testing:
        # https://github.com/python/cpython/blob/865c3b257fe38154a4320c7ee6afb416f665b9c2/Lib/test/string_tests.py#LL1049-L1099    # noqa: E501
        cpython_str = ['hello', 'helloworld', '']
        cpython_subs = [
            'he', 'hello', 'helloworld', 'ello',
            '', 'lowo', 'lo', 'he', 'lo', 'o',
        ]
        extra_subs = ['hellohellohello', ' ']
        for s in cpython_str + UNICODE_EXAMPLES:
            default_subs = ['', 'x', s[:-2], s[3:], s, s + s]
            for sub_str in cpython_subs + default_subs + extra_subs:
                for start in list(range(-20, 20)) + [None]:
                    for end in list(range(-20, 20)) + [None]:
                        msg = 'Results "{}".endswith("{}", {}, {})\
                               must be equal'
                        self.assertEqual(pyfunc(s, sub_str, start, end),
                                         cfunc(s, sub_str, start, end),
                                         msg=msg.format(s, sub_str, start, end))

    def test_endswith_tuple(self):
        pyfunc = endswith_usecase
        cfunc = njit(pyfunc)

        # Samples taken from CPython testing:
        # https://github.com/python/cpython/blob/865c3b257fe38154a4320c7ee6afb416f665b9c2/Lib/test/string_tests.py#L1049-L1099    # noqa: E501
        cpython_str = ['hello', 'helloworld', '']
        cpython_subs = [
            'he', 'hello', 'helloworld', 'ello',
            '', 'lowo', 'lo', 'he', 'lo', 'o',
        ]
        extra_subs = ['hellohellohello', ' ']
        for s in cpython_str + UNICODE_EXAMPLES:
            default_subs = ['', 'x', s[:-2], s[3:], s, s + s]
            for sub_str in cpython_subs + default_subs + extra_subs:
                msg = 'Results "{}".endswith({}) must be equal'
                tuple_subs = (sub_str, 'lo')
                self.assertEqual(pyfunc(s, tuple_subs),
                                 cfunc(s, tuple_subs),
                                 msg=msg.format(s, tuple_subs))

    def test_endswith_tuple_args(self):
        pyfunc = endswith_with_start_end_usecase
        cfunc = njit(pyfunc)

        # Samples taken from CPython testing:
        # https://github.com/python/cpython/blob/865c3b257fe38154a4320c7ee6afb416f665b9c2/Lib/test/string_tests.py#L1049-L1099    # noqa: E501
        cpython_str = ['hello', 'helloworld', '']
        cpython_subs = [
            'he', 'hello', 'helloworld', 'ello',
            '', 'lowo', 'lo', 'he', 'lo', 'o',
        ]
        extra_subs = ['hellohellohello', ' ']
        for s in cpython_str + UNICODE_EXAMPLES:
            default_subs = ['', 'x', s[:-2], s[3:], s, s + s]
            for sub_str in cpython_subs + default_subs + extra_subs:
                for start in list(range(-20, 20)) + [None]:
                    for end in list(range(-20, 20)) + [None]:
                        msg = 'Results "{}".endswith("{}", {}, {})\
                               must be equal'
                        tuple_subs = (sub_str, 'lo')
                        self.assertEqual(pyfunc(s, tuple_subs, start, end),
                                         cfunc(s, tuple_subs, start, end),
                                         msg=msg.format(s, tuple_subs,
                                                        start, end))

    def test_in(self, flags=no_pyobj_flags):
        pyfunc = in_usecase
        cfunc = njit(pyfunc)
        for a in UNICODE_EXAMPLES:
            extras = ['', 'xx', a[::-1], a[:-2], a[3:], a, a + a]
            for substr in extras:
                self.assertEqual(pyfunc(substr, a),
                                 cfunc(substr, a),
                                 "'%s' in '%s'?" % (substr, a))

    def test_partition_exception_invalid_sep(self):
        self.disable_leak_check()

        pyfunc = partition_usecase
        cfunc = njit(pyfunc)

        # Handle empty separator exception
        for func in [pyfunc, cfunc]:
            with self.assertRaises(ValueError) as raises:
                func('a', '')
            self.assertIn('empty separator', str(raises.exception))

        accepted_types = (types.UnicodeType, types.UnicodeCharSeq)
        with self.assertRaises(TypingError) as raises:
            cfunc('a', None)
        msg = '"sep" must be {}, not none'.format(accepted_types)
        self.assertIn(msg, str(raises.exception))

    def test_partition(self):
        pyfunc = partition_usecase
        cfunc = njit(pyfunc)

        CASES = [
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
        msg = 'Results of "{}".partition("{}") must be equal'
        for s, sep in CASES:
            self.assertEqual(pyfunc(s, sep), cfunc(s, sep),
                             msg=msg.format(s, sep))

    def test_find(self, flags=no_pyobj_flags):
        pyfunc = find_usecase
        cfunc = njit(pyfunc)

        default_subs = [
            (s, ['', 'xx', s[:-2], s[3:], s]) for s in UNICODE_EXAMPLES
        ]
        # Samples taken from CPython testing:
        # https://github.com/python/cpython/blob/201c8f79450628241574fba940e08107178dc3a5/Lib/test/test_unicode.py#L202-L231    # noqa: E501
        cpython_subs = [
            ('a' * 100 + '\u0102', ['\u0102', '\u0201', '\u0120', '\u0220']),
            ('a' * 100 + '\U00100304', ['\U00100304', '\U00100204',
                                        '\U00102004']),
            ('\u0102' * 100 + 'a', ['a']),
            ('\U00100304' * 100 + 'a', ['a']),
            ('\U00100304' * 100 + '\u0102', ['\u0102']),
            ('a' * 100, ['\u0102', '\U00100304', 'a\u0102', 'a\U00100304']),
            ('\u0102' * 100, ['\U00100304', '\u0102\U00100304']),
            ('\u0102' * 100 + 'a_', ['a_']),
            ('\U00100304' * 100 + 'a_', ['a_']),
            ('\U00100304' * 100 + '\u0102_', ['\u0102_']),
        ]
        for s, subs in default_subs + cpython_subs:
            for sub_str in subs:
                msg = 'Results "{}".find("{}") must be equal'
                self.assertEqual(pyfunc(s, sub_str), cfunc(s, sub_str),
                                 msg=msg.format(s, sub_str))

    def test_find_with_start_only(self):
        pyfunc = find_with_start_only_usecase
        cfunc = njit(pyfunc)

        for s in UNICODE_EXAMPLES:
            for sub_str in ['', 'xx', s[:-2], s[3:], s]:
                for start in list(range(-20, 20)) + [None]:
                    msg = 'Results "{}".find("{}", {}) must be equal'
                    self.assertEqual(pyfunc(s, sub_str, start),
                                     cfunc(s, sub_str, start),
                                     msg=msg.format(s, sub_str, start))

    def test_find_with_start_end(self):
        pyfunc = find_with_start_end_usecase
        cfunc = njit(pyfunc)

        starts = ends = list(range(-20, 20)) + [None]
        for s in UNICODE_EXAMPLES:
            for sub_str in ['', 'xx', s[:-2], s[3:], s]:
                for start, end in product(starts, ends):
                    msg = 'Results of "{}".find("{}", {}, {}) must be equal'
                    self.assertEqual(pyfunc(s, sub_str, start, end),
                                     cfunc(s, sub_str, start, end),
                                     msg=msg.format(s, sub_str, start, end))

    def test_find_exception_noninteger_start_end(self):
        pyfunc = find_with_start_end_usecase
        cfunc = njit(pyfunc)

        accepted = (types.Integer, types.NoneType)
        for start, end, name in [(0.1, 5, 'start'), (0, 0.5, 'end')]:
            with self.assertRaises(TypingError) as raises:
                cfunc('ascii', 'sci', start, end)
            msg = '"{}" must be {}, not float'.format(name, accepted)
            self.assertIn(msg, str(raises.exception))

    def test_rpartition_exception_invalid_sep(self):
        self.disable_leak_check()

        pyfunc = rpartition_usecase
        cfunc = njit(pyfunc)

        # Handle empty separator exception
        for func in [pyfunc, cfunc]:
            with self.assertRaises(ValueError) as raises:
                func('a', '')
            self.assertIn('empty separator', str(raises.exception))

        accepted_types = (types.UnicodeType, types.UnicodeCharSeq)
        with self.assertRaises(TypingError) as raises:
            cfunc('a', None)
        msg = '"sep" must be {}, not none'.format(accepted_types)
        self.assertIn(msg, str(raises.exception))

    def test_rpartition(self):
        pyfunc = rpartition_usecase
        cfunc = njit(pyfunc)

        CASES = [
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
        msg = 'Results of "{}".rpartition("{}") must be equal'
        for s, sep in CASES:
            self.assertEqual(pyfunc(s, sep), cfunc(s, sep),
                             msg=msg.format(s, sep))

    def test_count(self):
        pyfunc = count_usecase
        cfunc = njit(pyfunc)
        error_msg = "'{0}'.py_count('{1}') = {2}\n'{0}'.c_count('{1}') = {3}"

        for s, sub in UNICODE_COUNT_EXAMPLES:
            py_result = pyfunc(s, sub)
            c_result = cfunc(s, sub)
            self.assertEqual(py_result, c_result,
                             error_msg.format(s, sub, py_result, c_result))

    def test_count_with_start(self):
        pyfunc = count_with_start_usecase
        cfunc = njit(pyfunc)
        error_msg = "%s\n%s" % ("'{0}'.py_count('{1}', {2}) = {3}",
                                "'{0}'.c_count('{1}', {2}) = {4}")

        for s, sub in UNICODE_COUNT_EXAMPLES:
            for i in range(-18, 18):
                py_result = pyfunc(s, sub, i)
                c_result = cfunc(s, sub, i)
                self.assertEqual(py_result, c_result,
                                 error_msg.format(s, sub, i, py_result,
                                                  c_result))

            py_result = pyfunc(s, sub, None)
            c_result = cfunc(s, sub, None)
            self.assertEqual(py_result, c_result,
                             error_msg.format(s, sub, None, py_result,
                                              c_result))

    def test_count_with_start_end(self):
        pyfunc = count_with_start_end_usecase
        cfunc = njit(pyfunc)
        error_msg = "%s\n%s" % ("'{0}'.py_count('{1}', {2}, {3}) = {4}",
                                "'{0}'.c_count('{1}', {2}, {3}) = {5}")

        for s, sub in UNICODE_COUNT_EXAMPLES:
            for i, j in product(range(-18, 18), (-18, 18)):
                py_result = pyfunc(s, sub, i, j)
                c_result = cfunc(s, sub, i, j)
                self.assertEqual(py_result, c_result,
                                 error_msg.format(s, sub, i, j, py_result,
                                                  c_result))

            for j in range(-18, 18):
                py_result = pyfunc(s, sub, None, j)
                c_result = cfunc(s, sub, None, j)
                self.assertEqual(py_result, c_result,
                                 error_msg.format(s, sub, None, j, py_result,
                                                  c_result))

            py_result = pyfunc(s, sub, None, None)
            c_result = cfunc(s, sub, None, None)
            self.assertEqual(py_result, c_result,
                             error_msg.format(s, sub, None, None, py_result,
                                              c_result))

    def test_count_arg_type_check(self):
        cfunc = njit(count_with_start_end_usecase)

        with self.assertRaises(TypingError) as raises:
            cfunc('ascii', 'c', 1, 0.5)
        self.assertIn('The slice indices must be an Integer or None',
                      str(raises.exception))

        with self.assertRaises(TypingError) as raises:
            cfunc('ascii', 'c', 1.2, 7)
        self.assertIn('The slice indices must be an Integer or None',
                      str(raises.exception))

        with self.assertRaises(TypingError) as raises:
            cfunc('ascii', 12, 1, 7)
        self.assertIn('The substring must be a UnicodeType, not',
                      str(raises.exception))

    def test_count_optional_arg_type_check(self):
        pyfunc = count_with_start_end_usecase

        def try_compile_bad_optional(*args):
            bad_sig = types.int64(types.unicode_type,
                                  types.unicode_type,
                                  types.Optional(types.float64),
                                  types.Optional(types.float64))
            njit([bad_sig])(pyfunc)

        with self.assertRaises(TypingError) as raises:
            try_compile_bad_optional('tú quis?', 'tú', 1.1, 1.1)
        self.assertIn('The slice indices must be an Integer or None',
                      str(raises.exception))

        error_msg = "%s\n%s" % ("'{0}'.py_count('{1}', {2}, {3}) = {4}",
                                "'{0}'.c_count_op('{1}', {2}, {3}) = {5}")
        sig_optional = types.int64(types.unicode_type,
                                   types.unicode_type,
                                   types.Optional(types.int64),
                                   types.Optional(types.int64))
        cfunc_optional = njit([sig_optional])(pyfunc)

        py_result = pyfunc('tú quis?', 'tú', 0, 8)
        c_result = cfunc_optional('tú quis?', 'tú', 0, 8)
        self.assertEqual(py_result, c_result,
                         error_msg.format('tú quis?', 'tú', 0, 8, py_result,
                                          c_result))

    def test_rfind(self):
        pyfunc = rfind_usecase
        cfunc = njit(pyfunc)

        default_subs = [
            (s, ['', 'xx', s[:-2], s[3:], s]) for s in UNICODE_EXAMPLES
        ]
        # Samples taken from CPython testing:
        # https://github.com/python/cpython/blob/201c8f79450628241574fba940e08107178dc3a5/Lib/test/test_unicode.py#L233-L259    # noqa: E501
        cpython_subs = [
            ('\u0102' + 'a' * 100, ['\u0102', '\u0201', '\u0120', '\u0220']),
            ('\U00100304' + 'a' * 100, ['\U00100304', '\U00100204',
                                        '\U00102004']),
            ('abcdefghiabc', ['abc', '']),
            ('a' + '\u0102' * 100, ['a']),
            ('a' + '\U00100304' * 100, ['a']),
            ('\u0102' + '\U00100304' * 100, ['\u0102']),
            ('a' * 100, ['\u0102', '\U00100304', '\u0102a', '\U00100304a']),
            ('\u0102' * 100, ['\U00100304', '\U00100304\u0102']),
            ('_a' + '\u0102' * 100, ['_a']),
            ('_a' + '\U00100304' * 100, ['_a']),
            ('_\u0102' + '\U00100304' * 100, ['_\u0102']),
        ]
        for s, subs in default_subs + cpython_subs:
            for sub_str in subs:
                msg = 'Results "{}".rfind("{}") must be equal'
                self.assertEqual(pyfunc(s, sub_str), cfunc(s, sub_str),
                                 msg=msg.format(s, sub_str))

    def test_rfind_with_start_only(self):
        pyfunc = rfind_with_start_only_usecase
        cfunc = njit(pyfunc)

        for s in UNICODE_EXAMPLES:
            for sub_str in ['', 'xx', s[:-2], s[3:], s]:
                for start in list(range(-20, 20)) + [None]:
                    msg = 'Results "{}".rfind("{}", {}) must be equal'
                    self.assertEqual(pyfunc(s, sub_str, start),
                                     cfunc(s, sub_str, start),
                                     msg=msg.format(s, sub_str, start))

    def test_rfind_with_start_end(self):
        pyfunc = rfind_with_start_end_usecase
        cfunc = njit(pyfunc)

        starts = list(range(-20, 20)) + [None]
        ends = list(range(-20, 20)) + [None]
        for s in UNICODE_EXAMPLES:
            for sub_str in ['', 'xx', s[:-2], s[3:], s]:
                for start, end in product(starts, ends):
                    msg = 'Results of "{}".rfind("{}", {}, {}) must be equal'
                    self.assertEqual(pyfunc(s, sub_str, start, end),
                                     cfunc(s, sub_str, start, end),
                                     msg=msg.format(s, sub_str, start, end))

    def test_rfind_wrong_substr(self):
        cfunc = njit(rfind_usecase)

        for s in UNICODE_EXAMPLES:
            for sub_str in [None, 1, False]:
                with self.assertRaises(TypingError) as raises:
                    cfunc(s, sub_str)
                msg = 'must be {}'.format(types.UnicodeType)
                self.assertIn(msg, str(raises.exception))

    def test_rfind_wrong_start_end(self):
        cfunc = njit(rfind_with_start_end_usecase)

        accepted_types = (types.Integer, types.NoneType)
        for s in UNICODE_EXAMPLES:
            for sub_str in ['', 'xx', s[:-2], s[3:], s]:
                # test wrong start
                for start, end in product([0.1, False], [-1, 1]):
                    with self.assertRaises(TypingError) as raises:
                        cfunc(s, sub_str, start, end)
                    msg = '"start" must be {}'.format(accepted_types)
                    self.assertIn(msg, str(raises.exception))

                # test wrong end
                for start, end in product([-1, 1], [-0.1, True]):
                    with self.assertRaises(TypingError) as raises:
                        cfunc(s, sub_str, start, end)
                    msg = '"end" must be {}'.format(accepted_types)
                    self.assertIn(msg, str(raises.exception))

    def test_rfind_wrong_start_end_optional(self):
        s = UNICODE_EXAMPLES[0]
        sub_str = s[1:-1]
        accepted_types = (types.Integer, types.NoneType)
        msg = 'must be {}'.format(accepted_types)

        def try_compile_wrong_start_optional(*args):
            wrong_sig_optional = types.int64(types.unicode_type,
                                             types.unicode_type,
                                             types.Optional(types.float64),
                                             types.Optional(types.intp))
            njit([wrong_sig_optional])(rfind_with_start_end_usecase)

        with self.assertRaises(TypingError) as raises:
            try_compile_wrong_start_optional(s, sub_str, 0.1, 1)
        self.assertIn(msg, str(raises.exception))

        def try_compile_wrong_end_optional(*args):
            wrong_sig_optional = types.int64(types.unicode_type,
                                             types.unicode_type,
                                             types.Optional(types.intp),
                                             types.Optional(types.float64))
            njit([wrong_sig_optional])(rfind_with_start_end_usecase)

        with self.assertRaises(TypingError) as raises:
            try_compile_wrong_end_optional(s, sub_str, 1, 0.1)
        self.assertIn(msg, str(raises.exception))

    def test_rindex(self):
        pyfunc = rindex_usecase
        cfunc = njit(pyfunc)

        default_subs = [
            (s, ['', s[:-2], s[3:], s]) for s in UNICODE_EXAMPLES
        ]
        # Samples taken from CPython testing:
        # https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Lib/test/test_unicode.py#L284-L308    # noqa: E501
        cpython_subs = [
            ('abcdefghiabc', ['', 'def', 'abc']),
            ('a' + '\u0102' * 100, ['a']),
            ('a' + '\U00100304' * 100, ['a']),
            ('\u0102' + '\U00100304' * 100, ['\u0102']),
            ('_a' + '\u0102' * 100, ['_a']),
            ('_a' + '\U00100304' * 100, ['_a']),
            ('_\u0102' + '\U00100304' * 100, ['_\u0102'])
        ]
        for s, subs in default_subs + cpython_subs:
            for sub_str in subs:
                msg = 'Results "{}".rindex("{}") must be equal'
                self.assertEqual(pyfunc(s, sub_str), cfunc(s, sub_str),
                                 msg=msg.format(s, sub_str))

    def test_index(self):
        pyfunc = index_usecase
        cfunc = njit(pyfunc)

        default_subs = [
            (s, ['', s[:-2], s[3:], s]) for s in UNICODE_EXAMPLES
        ]
        # Samples taken from CPython testing:
        # https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Lib/test/test_unicode.py#L260-L282    # noqa: E501
        cpython_subs = [
            ('abcdefghiabc', ['', 'def', 'abc']),
            ('\u0102' * 100 + 'a', ['a']),
            ('\U00100304' * 100 + 'a', ['a']),
            ('\U00100304' * 100 + '\u0102', ['\u0102']),
            ('\u0102' * 100 + 'a_', ['a_']),
            ('\U00100304' * 100 + 'a_', ['a_']),
            ('\U00100304' * 100 + '\u0102_', ['\u0102_'])
        ]
        for s, subs in default_subs + cpython_subs:
            for sub_str in subs:
                msg = 'Results "{}".index("{}") must be equal'
                self.assertEqual(pyfunc(s, sub_str), cfunc(s, sub_str),
                                 msg=msg.format(s, sub_str))

    def test_index_rindex_with_start_only(self):
        pyfuncs = [index_with_start_only_usecase,
                   rindex_with_start_only_usecase]
        messages = ['Results "{}".index("{}", {}) must be equal',
                    'Results "{}".rindex("{}", {}) must be equal']
        unicode_examples = [
            'ascii',
            '12345',
            '1234567890',
            '¡Y tú quién te crees?',
            '大处着眼，小处着手。',
        ]
        for pyfunc, msg in zip(pyfuncs, messages):
            cfunc = njit(pyfunc)
            for s in unicode_examples:
                l = len(s)
                cases = [
                    ('', list(range(-10, l + 1))),
                    (s[:-2], [0] + list(range(-10, 1 - l))),
                    (s[3:], list(range(4)) + list(range(-10, 4 - l))),
                    (s, [0] + list(range(-10, 1 - l))),
                ]
                for sub_str, starts in cases:
                    for start in starts + [None]:
                        self.assertEqual(pyfunc(s, sub_str, start),
                                         cfunc(s, sub_str, start),
                                         msg=msg.format(s, sub_str, start))

    def test_index_rindex_with_start_end(self):
        pyfuncs = [index_with_start_end_usecase, rindex_with_start_end_usecase]
        messages = ['Results of "{}".index("{}", {}, {}) must be equal',
                    'Results of "{}".rindex("{}", {}, {}) must be equal']
        unicode_examples = [
            'ascii',
            '12345',
            '1234567890',
            '¡Y tú quién te crees?',
            '大处着眼，小处着手。',
        ]
        for pyfunc, msg in zip(pyfuncs, messages):
            cfunc = njit(pyfunc)
            for s in unicode_examples:
                l = len(s)
                cases = [
                    ('', list(range(-10, l + 1)), list(range(l, 10))),
                    (s[:-2], [0] + list(range(-10, 1 - l)),
                     [-2, -1] + list(range(l - 2, 10))),
                    (s[3:], list(range(4)) + list(range(-10, -1)),
                     list(range(l, 10))),
                    (s, [0] + list(range(-10, 1 - l)), list(range(l, 10))),
                ]
                for sub_str, starts, ends in cases:
                    for start, end in product(starts + [None], ends):
                        self.assertEqual(pyfunc(s, sub_str, start, end),
                                         cfunc(s, sub_str, start, end),
                                         msg=msg.format(s, sub_str, start, end))

    def test_index_rindex_exception_substring_not_found(self):
        self.disable_leak_check()

        unicode_examples = [
            'ascii',
            '12345',
            '1234567890',
            '¡Y tú quién te crees?',
            '大处着眼，小处着手。',
        ]
        pyfuncs = [index_with_start_end_usecase, rindex_with_start_end_usecase]
        for pyfunc in pyfuncs:
            cfunc = njit(pyfunc)
            for s in unicode_examples:
                l = len(s)
                cases = [
                    ('', list(range(l + 1, 10)), [l]),
                    (s[:-2], [0], list(range(l - 2))),
                    (s[3:], list(range(4, 10)), [l]),
                    (s, [None], list(range(l))),
                ]
                for sub_str, starts, ends in cases:
                    for start, end in product(starts, ends):
                        for func in [pyfunc, cfunc]:
                            with self.assertRaises(ValueError) as raises:
                                func(s, sub_str, start, end)
                            msg = 'substring not found'
                            self.assertIn(msg, str(raises.exception))

    def test_index_rindex_exception_noninteger_start_end(self):
        accepted = (types.Integer, types.NoneType)
        pyfuncs = [index_with_start_end_usecase, rindex_with_start_end_usecase]
        for pyfunc in pyfuncs:
            cfunc = njit(pyfunc)
            for start, end, name in [(0.1, 5, 'start'), (0, 0.5, 'end')]:
                with self.assertRaises(TypingError) as raises:
                    cfunc('ascii', 'sci', start, end)
                msg = '"{}" must be {}, not float'.format(name, accepted)
                self.assertIn(msg, str(raises.exception))

    def test_getitem(self):
        pyfunc = getitem_usecase
        cfunc = njit(pyfunc)

        for s in UNICODE_EXAMPLES:
            for i in range(-len(s), len(s)):
                self.assertEqual(pyfunc(s, i),
                                 cfunc(s, i),
                                 "'%s'[%d]?" % (s, i))

    def test_getitem_scalar_kind(self):
        # See issue #6135, make sure that getitem returns a char of the minimal
        # kind required to represent the "got" item, this is done via the use
        # of `hash` in the test function as it is sensitive to kind.
        pyfunc = getitem_check_kind_usecase
        cfunc = njit(pyfunc)
        samples = ['a\u1234', '¡着']
        for s in samples:
            for i in range(-len(s), len(s)):
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

    def test_getitem_slice2_kind(self):
        # See issue #6135. Also see note in test_getitem_scalar_kind regarding
        # testing.
        pyfunc = getitem_check_kind_usecase
        cfunc = njit(pyfunc)
        samples = ['abc\u1234\u1234', '¡¡¡着着着']
        for s in samples:
            for i in [-2, -1, 0, 1, 2, len(s), len(s) + 1]:
                for j in [-2, -1, 0, 1, 2, len(s), len(s) + 1]:
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

    def test_getitem_slice3_kind(self):
        # See issue #6135. Also see note in test_getitem_scalar_kind regarding
        # testing.
        pyfunc = getitem_check_kind_usecase
        cfunc = njit(pyfunc)
        samples = ['abc\u1234\u1234',
                   'a\u1234b\u1234c'
                   '¡¡¡着着着',
                   '¡着¡着¡着',
                   '着a着b着c',
                   '¡着a¡着b¡着c',
                   '¡着a着¡c',]
        for s in samples:
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

    def test_slice_ascii_flag(self):
        """
        Make sure ascii flag is False when ascii and non-ascii characters are
        mixed in output of Unicode slicing.
        """
        @njit
        def f(s):
            return s[::2]._is_ascii, s[1::2]._is_ascii

        s = "¿abc¡Y tú, quién te cre\t\tes?"
        self.assertEqual(f(s), (0, 1))

    def test_zfill(self):
        pyfunc = zfill_usecase
        cfunc = njit(pyfunc)

        ZFILL_INPUTS = [
            'ascii',
            '+ascii',
            '-ascii',
            '-asc ii-',
            '12345',
            '-12345',
            '+12345',
            '',
            '¡Y tú crs?',
            '🐍⚡',
            '+🐍⚡',
            '-🐍⚡',
            '大眼，小手。',
            '+大眼，小手。',
            '-大眼，小手。',
        ]

        with self.assertRaises(TypingError) as raises:
            cfunc(ZFILL_INPUTS[0], 1.1)
        self.assertIn('<width> must be an Integer', str(raises.exception))

        for s in ZFILL_INPUTS:
            for width in range(-3, 20):
                self.assertEqual(pyfunc(s, width),
                                 cfunc(s, width))

    def test_concat(self, flags=no_pyobj_flags):
        pyfunc = concat_usecase
        cfunc = njit(pyfunc)
        for a in UNICODE_EXAMPLES:
            for b in UNICODE_EXAMPLES[::-1]:
                self.assertEqual(pyfunc(a, b),
                                 cfunc(a, b),
                                 "'%s' + '%s'?" % (a, b))

    def test_repeat(self, flags=no_pyobj_flags):
        pyfunc = repeat_usecase
        cfunc = njit(pyfunc)
        for a in UNICODE_EXAMPLES:
            for b in (-1, 0, 1, 2, 3, 4, 5, 7, 8, 15, 70):
                self.assertEqual(pyfunc(a, b),
                                 cfunc(a, b))
                self.assertEqual(pyfunc(b, a),
                                 cfunc(b, a))

    def test_repeat_exception_float(self):
        self.disable_leak_check()
        cfunc = njit(repeat_usecase)
        with self.assertRaises(TypingError) as raises:
            cfunc('hi', 2.5)
        self.assertIn(_header_lead + ' Function(<built-in function mul>)',
                      str(raises.exception))

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

        for pyfunc, fmt_str in [(split_with_maxsplit_usecase,
                                 "'%s'.split('%s', %d)?"),
                                (split_with_maxsplit_kwarg_usecase,
                                 "'%s'.split('%s', maxsplit=%d)?")]:

            cfunc = njit(pyfunc)
            for test_str, splitter, maxsplit in CASES:
                self.assertEqual(pyfunc(test_str, splitter, maxsplit),
                                 cfunc(test_str, splitter, maxsplit),
                                 fmt_str % (test_str, splitter, maxsplit))

    def test_split_whitespace(self):
        # explicit sep=None cases covered in test_split and
        # test_split_with_maxsplit
        pyfunc = split_whitespace_usecase
        cfunc = njit(pyfunc)

        # list copied from
        # https://github.com/python/cpython/blob/master/Objects/unicodetype_db.h
        all_whitespace = ''.join(map(chr, [
            0x0009, 0x000A, 0x000B, 0x000C, 0x000D, 0x001C, 0x001D, 0x001E,
            0x001F, 0x0020, 0x0085, 0x00A0, 0x1680, 0x2000, 0x2001, 0x2002,
            0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200A,
            0x2028, 0x2029, 0x202F, 0x205F, 0x3000
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

    def test_split_exception_invalid_keepends(self):
        pyfunc = splitlines_with_keepends_usecase
        cfunc = njit(pyfunc)

        accepted_types = (types.Integer, int, types.Boolean, bool)
        for ty, keepends in (('none', None), ('unicode_type', 'None')):
            with self.assertRaises(TypingError) as raises:
                cfunc('\n', keepends)
            msg = '"keepends" must be {}, not {}'.format(accepted_types, ty)
            self.assertIn(msg, str(raises.exception))

    def test_splitlines(self):
        pyfunc = splitlines_usecase
        cfunc = njit(pyfunc)

        cases = ['', '\n', 'abc\r\rabc\r\n', '🐍⚡\v', '\f🐍⚡\f\v\v🐍\x85',
                 '\u2028aba\u2029baba', '\n\r\na\v\fb\x0b\x0cc\x1c\x1d\x1e']

        msg = 'Results of "{}".splitlines() must be equal'
        for s in cases:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

    def test_splitlines_with_keepends(self):
        pyfuncs = [
            splitlines_with_keepends_usecase,
            splitlines_with_keepends_kwarg_usecase
        ]
        messages = [
            'Results of "{}".splitlines({}) must be equal',
            'Results of "{}".splitlines(keepends={}) must be equal'
        ]
        cases = ['', '\n', 'abc\r\rabc\r\n', '🐍⚡\v', '\f🐍⚡\f\v\v🐍\x85',
                 '\u2028aba\u2029baba', '\n\r\na\v\fb\x0b\x0cc\x1c\x1d\x1e']
        all_keepends = [True, False, 0, 1, -1, 100]

        for pyfunc, msg in zip(pyfuncs, messages):
            cfunc = njit(pyfunc)
            for s, keepends in product(cases, all_keepends):
                self.assertEqual(pyfunc(s, keepends), cfunc(s, keepends),
                                 msg=msg.format(s, keepends))

    def test_rsplit_exception_empty_sep(self):
        self.disable_leak_check()

        pyfunc = rsplit_usecase
        cfunc = njit(pyfunc)

        # Handle empty separator exception
        for func in [pyfunc, cfunc]:
            with self.assertRaises(ValueError) as raises:
                func('a', '')
            self.assertIn('empty separator', str(raises.exception))

    def test_rsplit_exception_noninteger_maxsplit(self):
        pyfunc = rsplit_with_maxsplit_usecase
        cfunc = njit(pyfunc)

        accepted_types = (types.Integer, int)
        for sep in [' ', None]:
            with self.assertRaises(TypingError) as raises:
                cfunc('a', sep, 2.4)
            msg = '"maxsplit" must be {}, not float'.format(accepted_types)
            self.assertIn(msg, str(raises.exception))

    def test_rsplit(self):
        pyfunc = rsplit_usecase
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
        msg = 'Results of "{}".rsplit("{}") must be equal'
        for s, sep in CASES:
            self.assertEqual(pyfunc(s, sep), cfunc(s, sep),
                             msg=msg.format(s, sep))

    def test_rsplit_with_maxsplit(self):
        pyfuncs = [rsplit_with_maxsplit_usecase,
                   rsplit_with_maxsplit_kwarg_usecase]
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
        messages = [
            'Results of "{}".rsplit("{}", {}) must be equal',
            'Results of "{}".rsplit("{}", maxsplit={}) must be equal'
        ]

        for pyfunc, msg in zip(pyfuncs, messages):
            cfunc = njit(pyfunc)
            for test_str, sep, maxsplit in CASES:
                self.assertEqual(pyfunc(test_str, sep, maxsplit),
                                 cfunc(test_str, sep, maxsplit),
                                 msg=msg.format(test_str, sep, maxsplit))

    def test_rsplit_whitespace(self):
        pyfunc = rsplit_whitespace_usecase
        cfunc = njit(pyfunc)

        # list copied from
        # https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Objects/unicodetype_db.h#L5996-L6031    # noqa: E501
        all_whitespace = ''.join(map(chr, [
            0x0009, 0x000A, 0x000B, 0x000C, 0x000D, 0x001C, 0x001D, 0x001E,
            0x001F, 0x0020, 0x0085, 0x00A0, 0x1680, 0x2000, 0x2001, 0x2002,
            0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200A,
            0x2028, 0x2029, 0x202F, 0x205F, 0x3000
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
        msg = 'Results of "{}".rsplit() must be equal'
        for s in CASES:
            self.assertEqual(pyfunc(s), cfunc(s), msg.format(s))

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
            cfunc('', [1, 2, 3])
        # This error message is obscure, but indicates the error was trapped
        # in the typing of str.join()
        # Feel free to change this as we update error messages.
        exc_message = str(raises.exception)
        self.assertIn(
            "During: resolving callee type: BoundFunction",
            exc_message,
        )
        # could be int32 or int64
        self.assertIn("reflected list(int", exc_message)

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

    def test_justification(self):
        for pyfunc, case_name in [(center_usecase, 'center'),
                                  (ljust_usecase, 'ljust'),
                                  (rjust_usecase, 'rjust')]:
            cfunc = njit(pyfunc)

            with self.assertRaises(TypingError) as raises:
                cfunc(UNICODE_EXAMPLES[0], 1.1)
            self.assertIn('The width must be an Integer', str(raises.exception))

            for s in UNICODE_EXAMPLES:
                for width in range(-3, 20):
                    self.assertEqual(pyfunc(s, width),
                                     cfunc(s, width),
                                     "'%s'.%s(%d)?" % (s, case_name, width))

    def test_justification_fillchar(self):
        for pyfunc, case_name in [(center_usecase_fillchar, 'center'),
                                  (ljust_usecase_fillchar, 'ljust'),
                                  (rjust_usecase_fillchar, 'rjust')]:
            cfunc = njit(pyfunc)

            # allowed fillchar cases
            for fillchar in [' ', '+', 'ú', '处']:
                with self.assertRaises(TypingError) as raises:
                    cfunc(UNICODE_EXAMPLES[0], 1.1, fillchar)
                self.assertIn('The width must be an Integer',
                              str(raises.exception))

                for s in UNICODE_EXAMPLES:
                    for width in range(-3, 20):
                        self.assertEqual(pyfunc(s, width, fillchar),
                                         cfunc(s, width, fillchar),
                                         "'%s'.%s(%d, '%s')?" % (s, case_name,
                                                                 width,
                                                                 fillchar))

    def test_justification_fillchar_exception(self):
        self.disable_leak_check()

        for pyfunc in [center_usecase_fillchar,
                       ljust_usecase_fillchar,
                       rjust_usecase_fillchar]:
            cfunc = njit(pyfunc)

            # disallowed fillchar cases
            for fillchar in ['', '+0', 'quién', '处着']:
                with self.assertRaises(ValueError) as raises:
                    cfunc(UNICODE_EXAMPLES[0], 20, fillchar)
                self.assertIn('The fill character must be exactly one',
                              str(raises.exception))

            # forbid fillchar cases with different types
            for fillchar in [1, 1.1]:
                with self.assertRaises(TypingError) as raises:
                    cfunc(UNICODE_EXAMPLES[0], 20, fillchar)
                self.assertIn('The fillchar must be a UnicodeType',
                              str(raises.exception))

    def test_inplace_concat(self, flags=no_pyobj_flags):
        pyfunc = inplace_concat_usecase
        cfunc = njit(pyfunc)
        for a in UNICODE_EXAMPLES:
            for b in UNICODE_EXAMPLES[::-1]:
                self.assertEqual(pyfunc(a, b),
                                 cfunc(a, b),
                                 "'%s' + '%s'?" % (a, b))

    def test_isidentifier(self):
        def pyfunc(s):
            return s.isidentifier()

        cfunc = njit(pyfunc)
        # https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Lib/test/test_unicode.py#L695-L708    # noqa: E501
        cpython = ['a', 'Z', '_', 'b0', 'bc', 'b_', 'µ',
                   '𝔘𝔫𝔦𝔠𝔬𝔡𝔢', ' ', '[', '©', '0']
        # https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Lib/test/test_unicode.py#L742-L749    # noqa: E501
        cpython_extras = ['\uD800', '\uDFFF', '\uD800\uD800', '\uDFFF\uDFFF',
                          'a\uD800b\uDFFF', 'a\uDFFFb\uD800',
                          'a\uD800b\uDFFFa', 'a\uDFFFb\uD800a']

        msg = 'Results of "{}".isidentifier() must be equal'
        for s in UNICODE_EXAMPLES + [''] + cpython + cpython_extras:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

    def test_strip(self):

        STRIP_CASES = [
            ('ass cii', 'ai'),
            ('ass cii', None),
            ('asscii', 'ai '),
            ('asscii ', 'ai '),
            (' asscii  ', 'ai '),
            (' asscii  ', 'asci '),
            (' asscii  ', 's'),
            ('      ', ' '),
            ('', ' '),
            ('', ''),
            ('', None),
            (' ', None),
            ('  asscii  ', 'ai '),
            ('  asscii  ', ''),
            ('  asscii  ', None),
            ('tú quién te crees?', 'étú? '),
            ('  tú quién te crees?   ', 'étú? '),
            ('  tú qrees?   ', ''),
            ('  tú quién te crees?   ', None),
            ('大处 着眼，小处着手。大大大处', '大处'),
            (' 大处大处  ', ''),
            ('\t\nabcd\t', '\ta'),
            (' 大处大处  ', None),
            ('\t abcd \t', None),
            ('\n abcd \n', None),
            ('\r abcd \r', None),
            ('\x0b abcd \x0b', None),
            ('\x0c abcd \x0c', None),
            ('\u2029abcd\u205F', None),
            ('\u0085abcd\u2009', None)
        ]

        # form with no parameter
        for pyfunc, case_name in [(strip_usecase, 'strip'),
                                  (lstrip_usecase, 'lstrip'),
                                  (rstrip_usecase, 'rstrip')]:
            cfunc = njit(pyfunc)

            for string, chars in STRIP_CASES:
                self.assertEqual(pyfunc(string),
                                 cfunc(string),
                                 "'%s'.%s()?" % (string, case_name))
        # parametrized form
        for pyfunc, case_name in [(strip_usecase_chars, 'strip'),
                                  (lstrip_usecase_chars, 'lstrip'),
                                  (rstrip_usecase_chars, 'rstrip')]:
            cfunc = njit(pyfunc)

            sig1 = types.unicode_type(types.unicode_type,
                                      types.Optional(types.unicode_type))
            cfunc_optional = njit([sig1])(pyfunc)

            def try_compile_bad_optional(*args):
                bad = types.unicode_type(types.unicode_type,
                                         types.Optional(types.float64))
                njit([bad])(pyfunc)

            for fn in cfunc, try_compile_bad_optional:
                with self.assertRaises(TypingError) as raises:
                    fn('tú quis?', 1.1)
                self.assertIn('The arg must be a UnicodeType or None',
                              str(raises.exception))

            for fn in cfunc, cfunc_optional:

                for string, chars in STRIP_CASES:
                    self.assertEqual(pyfunc(string, chars),
                                     fn(string, chars),
                                     "'%s'.%s('%s')?" % (string, case_name,
                                                         chars))

    def test_isspace(self):
        def pyfunc(s):
            return s.isspace()

        cfunc = njit(pyfunc)
        # https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Lib/test/test_unicode.py#L613-L621    # noqa: E501
        cpython = ['\u2000', '\u200a', '\u2014', '\U00010401', '\U00010427',
                   '\U00010429', '\U0001044E', '\U0001F40D', '\U0001F46F']
        # https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Lib/test/test_unicode.py#L742-L749    # noqa: E501
        cpython_extras = ['\uD800', '\uDFFF', '\uD800\uD800', '\uDFFF\uDFFF',
                          'a\uD800b\uDFFF', 'a\uDFFFb\uD800',
                          'a\uD800b\uDFFFa', 'a\uDFFFb\uD800a']

        msg = 'Results of "{}".isspace() must be equal'
        for s in UNICODE_EXAMPLES + [''] + cpython + cpython_extras:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

    def test_istitle(self):
        pyfunc = istitle_usecase
        cfunc = njit(pyfunc)
        error_msg = "'{0}'.py_istitle() = {1}\n'{0}'.c_istitle() = {2}"

        unicode_title = [x.title() for x in UNICODE_EXAMPLES]
        special = [
            '',
            '    ',
            '  AA  ',
            '  Ab  ',
            '1',
            'A123',
            'A12Bcd',
            '+abA',
            '12Abc',
            'A12abc',
            '%^Abc 5 $% Def'
            '𐐁𐐩',
            '𐐧𐑎',
            '𐐩',
            '𐑎',
            '🐍 Is',
            '🐍 NOT',
            '👯Is',
            'ῼ',
            'Greek ῼitlecases ...'
        ]
        ISTITLE_EXAMPLES = UNICODE_EXAMPLES + unicode_title + special

        for s in ISTITLE_EXAMPLES:
            py_result = pyfunc(s)
            c_result = cfunc(s)
            self.assertEqual(py_result, c_result,
                             error_msg.format(s, py_result, c_result))

    def test_isprintable(self):
        def pyfunc(s):
            return s.isprintable()

        cfunc = njit(pyfunc)
        # https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Lib/test/test_unicode.py#L710-L723    # noqa: E501
        cpython = ['', ' ', 'abcdefg', 'abcdefg\n', '\u0374', '\u0378',
                   '\ud800', '\U0001F46F', '\U000E0020']

        msg = 'Results of "{}".isprintable() must be equal'
        for s in UNICODE_EXAMPLES + cpython:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

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

    def test_not(self):
        def pyfunc(x):
            return not x

        cfunc = njit(pyfunc)
        for a in UNICODE_EXAMPLES:
            args = [a]
            self.assertEqual(pyfunc(*args), cfunc(*args),
                             msg='failed on {}'.format(args))

    def test_capitalize(self):
        def pyfunc(x):
            return x.capitalize()

        cfunc = njit(pyfunc)
        # Samples taken from CPython testing:
        # https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Lib/test/test_unicode.py#L800-L815    # noqa: E501
        cpython = ['\U0001044F', '\U0001044F\U0001044F', '\U00010427\U0001044F',
                   '\U0001044F\U00010427', 'X\U00010427x\U0001044F', 'h\u0130',
                   '\u1fd2\u0130', 'ﬁnnish', 'A\u0345\u03a3']
        # https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Lib/test/test_unicode.py#L926    # noqa: E501
        cpython_extras = ['\U00010000\U00100000']

        msg = 'Results of "{}".capitalize() must be equal'
        for s in UNICODE_EXAMPLES + [''] + cpython + cpython_extras:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

    def test_isupper(self):
        def pyfunc(x):
            return x.isupper()

        cfunc = njit(pyfunc)
        uppers = [x.upper() for x in UNICODE_EXAMPLES]
        extras = ["AA12A", "aa12a", "大AA12A", "大aa12a", "AAAǄA", "A 1 1 大"]

        # Samples taken from CPython testing:
        # https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Lib/test/test_unicode.py#L585-L599    # noqa: E501
        cpython = ['\u2167', '\u2177', '\U00010401', '\U00010427', '\U00010429',
                   '\U0001044E', '\U0001F40D', '\U0001F46F']
        fourxcpy = [x * 4 for x in cpython]

        for a in UNICODE_EXAMPLES + uppers + extras + cpython + fourxcpy:
            args = [a]
            self.assertEqual(pyfunc(*args), cfunc(*args),
                             msg='failed on {}'.format(args))

    def test_upper(self):
        def pyfunc(x):
            return x.upper()

        cfunc = njit(pyfunc)
        for a in UNICODE_EXAMPLES:
            args = [a]
            self.assertEqual(pyfunc(*args), cfunc(*args),
                             msg='failed on {}'.format(args))

    def test_casefold(self):
        def pyfunc(x):
            return x.casefold()

        cfunc = njit(pyfunc)
        # https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Lib/test/test_unicode.py#L774-L781    # noqa: E501
        cpython = ['hello', 'hELlo', 'ß', 'ﬁ', '\u03a3',
                   'A\u0345\u03a3', '\u00b5']
        # https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Lib/test/test_unicode.py#L924    # noqa: E501
        cpython_extras = ['\U00010000\U00100000']

        msg = 'Results of "{}".casefold() must be equal'
        for s in UNICODE_EXAMPLES + [''] + cpython + cpython_extras:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

    def test_isalpha(self):
        def pyfunc(x):
            return x.isalpha()

        cfunc = njit(pyfunc)
        # Samples taken from CPython testing:
        # https://github.com/python/cpython/blob/201c8f79450628241574fba940e08107178dc3a5/Lib/test/test_unicode.py#L630-L640    # noqa: E501
        cpython = ['\u1FFc', '\U00010401', '\U00010427', '\U00010429',
                   '\U0001044E', '\U0001F40D', '\U0001F46F']
        # https://github.com/python/cpython/blob/201c8f79450628241574fba940e08107178dc3a5/Lib/test/test_unicode.py#L738-L745    # noqa: E501
        extras = ['\uD800', '\uDFFF', '\uD800\uD800', '\uDFFF\uDFFF',
                  'a\uD800b\uDFFF', 'a\uDFFFb\uD800',
                  'a\uD800b\uDFFFa', 'a\uDFFFb\uD800a']

        msg = 'Results of "{}".isalpha() must be equal'
        for s in UNICODE_EXAMPLES + [''] + extras + cpython:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

    def test_isascii(self):
        def pyfunc(x):
            return x.isascii()

        cfunc = njit(pyfunc)
        # Samples taken from CPython testing:
        # https://github.com/python/cpython/blob/865c3b257fe38154a4320c7ee6afb416f665b9c2/Lib/test/string_tests.py#L913-L926     # noqa: E501
        cpython = ['', '\x00', '\x7f', '\x00\x7f', '\x80', '\xe9', ' ']

        msg = 'Results of "{}".isascii() must be equal'
        for s in UNICODE_EXAMPLES + cpython:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

    def test_title(self):
        pyfunc = title
        cfunc = njit(pyfunc)
        # Samples taken from CPython testing:
        # https://github.com/python/cpython/blob/201c8f79450628241574fba940e08107178dc3a5/Lib/test/test_unicode.py#L813-L828    # noqa: E501
        cpython = ['\U0001044F', '\U0001044F\U0001044F',
                   '\U0001044F\U0001044F \U0001044F\U0001044F',
                   '\U00010427\U0001044F \U00010427\U0001044F',
                   '\U0001044F\U00010427 \U0001044F\U00010427',
                   'X\U00010427x\U0001044F X\U00010427x\U0001044F',
                   'ﬁNNISH', 'A\u03a3 \u1fa1xy', 'A\u03a3A']

        msg = 'Results of "{}".title() must be equal'
        for s in UNICODE_EXAMPLES + [''] + cpython:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

    def test_swapcase(self):
        def pyfunc(x):
            return x.swapcase()

        cfunc = njit(pyfunc)
        # https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Lib/test/test_unicode.py#L834-L858    # noqa: E501
        cpython = ['\U0001044F', '\U00010427', '\U0001044F\U0001044F',
                   '\U00010427\U0001044F', '\U0001044F\U00010427',
                   'X\U00010427x\U0001044F', 'ﬁ', '\u0130', '\u03a3',
                   '\u0345\u03a3', 'A\u0345\u03a3', 'A\u0345\u03a3a',
                   'A\u0345\u03a3', 'A\u03a3\u0345', '\u03a3\u0345 ',
                   '\u03a3', 'ß', '\u1fd2']
        # https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Lib/test/test_unicode.py#L928    # noqa: E501
        cpython_extras = ['\U00010000\U00100000']

        msg = 'Results of "{}".swapcase() must be equal'
        for s in UNICODE_EXAMPLES + [''] + cpython + cpython_extras:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

    def test_islower(self):
        pyfunc = islower_usecase
        cfunc = njit(pyfunc)
        lowers = [x.lower() for x in UNICODE_EXAMPLES]
        extras = ['AA12A', 'aa12a', '大AA12A', '大aa12a', 'AAAǄA', 'A 1 1 大']

        # Samples taken from CPython testing:
        # https://github.com/python/cpython/blob/201c8f79450628241574fba940e08107178dc3a5/Lib/test/test_unicode.py#L586-L600    # noqa: E501
        cpython = ['\u2167', '\u2177', '\U00010401', '\U00010427',
                   '\U00010429', '\U0001044E', '\U0001F40D', '\U0001F46F']
        cpython += [x * 4 for x in cpython]

        msg = 'Results of "{}".islower() must be equal'
        for s in UNICODE_EXAMPLES + lowers + [''] + extras + cpython:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

    def test_isalnum(self):
        def pyfunc(x):
            return x.isalnum()

        cfunc = njit(pyfunc)
        # Samples taken from CPython testing:
        # https://github.com/python/cpython/blob/201c8f79450628241574fba940e08107178dc3a5/Lib/test/test_unicode.py#L624-L628    # noqa: E501
        cpython = ['\U00010401', '\U00010427', '\U00010429', '\U0001044E',
                   '\U0001D7F6', '\U00011066', '\U000104A0', '\U0001F107']
        # https://github.com/python/cpython/blob/201c8f79450628241574fba940e08107178dc3a5/Lib/test/test_unicode.py#L738-L745    # noqa: E501
        extras = ['\uD800', '\uDFFF', '\uD800\uD800', '\uDFFF\uDFFF',
                  'a\uD800b\uDFFF', 'a\uDFFFb\uD800',
                  'a\uD800b\uDFFFa', 'a\uDFFFb\uD800a']

        msg = 'Results of "{}".isalnum() must be equal'
        for s in UNICODE_EXAMPLES + [''] + extras + cpython:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

    def test_lower(self):
        pyfunc = lower_usecase
        cfunc = njit(pyfunc)
        extras = ['AA12A', 'aa12a', '大AA12A', '大aa12a', 'AAAǄA', 'A 1 1 大']

        # Samples taken from CPython testing:
        # https://github.com/python/cpython/blob/201c8f79450628241574fba940e08107178dc3a5/Lib/test/test_unicode.py#L748-L758    # noqa: E501
        cpython = ['\U00010401', '\U00010427', '\U0001044E', '\U0001F46F',
                   '\U00010427\U00010427', '\U00010427\U0001044F',
                   'X\U00010427x\U0001044F', '\u0130']

        # special cases for sigma from CPython testing:
        # https://github.com/python/cpython/blob/201c8f79450628241574fba940e08107178dc3a5/Lib/test/test_unicode.py#L759-L768    # noqa: E501
        sigma = ['\u03a3', '\u0345\u03a3', 'A\u0345\u03a3', 'A\u0345\u03a3a',
                 '\u03a3\u0345 ', '\U0008fffe', '\u2177']

        extra_sigma = 'A\u03a3\u03a2'
        sigma.append(extra_sigma)

        msg = 'Results of "{}".lower() must be equal'
        for s in UNICODE_EXAMPLES + [''] + extras + cpython + sigma:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

    def test_isnumeric(self):
        def pyfunc(x):
            return x.isnumeric()

        cfunc = njit(pyfunc)
        # https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Lib/test/test_unicode.py#L676-L693    # noqa: E501
        cpython = ['', 'a', '0', '\u2460', '\xbc', '\u0660', '0123456789',
                   '0123456789a', '\U00010401', '\U00010427', '\U00010429',
                   '\U0001044E', '\U0001F40D', '\U0001F46F', '\U00011065',
                   '\U0001D7F6', '\U00011066', '\U000104A0', '\U0001F107']
        # https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Lib/test/test_unicode.py#L742-L749    # noqa: E501
        cpython_extras = ['\uD800', '\uDFFF', '\uD800\uD800', '\uDFFF\uDFFF',
                          'a\uD800b\uDFFF', 'a\uDFFFb\uD800', 'a\uD800b\uDFFFa',
                          'a\uDFFFb\uD800a']

        msg = 'Results of "{}".isnumeric() must be equal'
        for s in UNICODE_EXAMPLES + [''] + cpython + cpython_extras:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

    def test_isdigit(self):
        def pyfunc(x):
            return x.isdigit()

        cfunc = njit(pyfunc)
        # https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Lib/test/test_unicode.py#L664-L674    # noqa: E501
        cpython = ['\u2460', '\xbc', '\u0660', '\U00010401', '\U00010427',
                   '\U00010429', '\U0001044E', '\U0001F40D', '\U0001F46F',
                   '\U00011065', '\U0001D7F6', '\U00011066', '\U000104A0',
                   '\U0001F107']
        # https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Lib/test/test_unicode.py#L742-L749    # noqa: E501
        cpython_extras = ['\uD800', '\uDFFF', '\uD800\uD800', '\uDFFF\uDFFF',
                          'a\uD800b\uDFFF', 'a\uDFFFb\uD800',
                          'a\uD800b\uDFFFa', 'a\uDFFFb\uD800a']

        msg = 'Results of "{}".isdigit() must be equal'
        for s in UNICODE_EXAMPLES + [''] + cpython + cpython_extras:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

    def test_isdecimal(self):
        def pyfunc(x):
            return x.isdecimal()

        cfunc = njit(pyfunc)
        # https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Lib/test/test_unicode.py#L646-L662    # noqa: E501
        cpython = ['', 'a', '0', '\u2460', '\xbc', '\u0660', '0123456789',
                   '0123456789a', '\U00010401', '\U00010427', '\U00010429',
                   '\U0001044E', '\U0001F40D', '\U0001F46F', '\U00011065',
                   '\U0001F107', '\U0001D7F6', '\U00011066', '\U000104A0']
        # https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Lib/test/test_unicode.py#L742-L749    # noqa: E501
        cpython_extras = ['\uD800', '\uDFFF', '\uD800\uD800', '\uDFFF\uDFFF',
                          'a\uD800b\uDFFF', 'a\uDFFFb\uD800', 'a\uD800b\uDFFFa',
                          'a\uDFFFb\uD800a']

        msg = 'Results of "{}".isdecimal() must be equal'
        for s in UNICODE_EXAMPLES + [''] + cpython + cpython_extras:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

    def test_replace(self):
        pyfunc = replace_usecase
        cfunc = njit(pyfunc)

        CASES = [
            ('abc', '', 'A'),
            ('', '⚡', 'A'),
            ('abcabc', '⚡', 'A'),
            ('🐍⚡', '⚡', 'A'),
            ('🐍⚡🐍', '⚡', 'A'),
            ('abababa', 'a', 'A'),
            ('abababa', 'b', 'A'),
            ('abababa', 'c', 'A'),
            ('abababa', 'ab', 'A'),
            ('abababa', 'aba', 'A'),
        ]

        for test_str, old_str, new_str in CASES:
            self.assertEqual(pyfunc(test_str, old_str, new_str),
                             cfunc(test_str, old_str, new_str),
                             "'%s'.replace('%s', '%s')?" %
                             (test_str, old_str, new_str))

    def test_replace_with_count(self):
        pyfunc = replace_with_count_usecase
        cfunc = njit(pyfunc)

        CASES = [
            ('abc', '', 'A'),
            ('', '⚡', 'A'),
            ('abcabc', '⚡', 'A'),
            ('🐍⚡', '⚡', 'A'),
            ('🐍⚡🐍', '⚡', 'A'),
            ('abababa', 'a', 'A'),
            ('abababa', 'b', 'A'),
            ('abababa', 'c', 'A'),
            ('abababa', 'ab', 'A'),
            ('abababa', 'aba', 'A'),
        ]

        count_test = [-1, 1, 0, 5]

        for test_str, old_str, new_str in CASES:
            for count in count_test:
                self.assertEqual(pyfunc(test_str, old_str, new_str, count),
                                 cfunc(test_str, old_str, new_str, count),
                                 "'%s'.replace('%s', '%s', '%s')?" %
                                 (test_str, old_str, new_str, count))

    def test_replace_unsupported(self):
        def pyfunc(s, x, y, count):
            return s.replace(x, y, count)

        cfunc = njit(pyfunc)

        with self.assertRaises(TypingError) as raises:
            cfunc('ababababab', 'ba', 'qqq', 3.5)
        msg = 'Unsupported parameters. The parameters must be Integer.'
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(TypingError) as raises:
            cfunc('ababababab', 0, 'qqq', 3)
        msg = 'The object must be a UnicodeType.'
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(TypingError) as raises:
            cfunc('ababababab', 'ba', 0, 3)
        msg = 'The object must be a UnicodeType.'
        self.assertIn(msg, str(raises.exception))


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

    def test_ascii_flag_unbox(self):
        @njit
        def f(s):
            return s._is_ascii

        for s in UNICODE_EXAMPLES:
            self.assertEqual(f(s), isascii(s))

    def test_ascii_flag_join(self):
        @njit
        def f():
            s1 = 'abc'
            s2 = '123'
            s3 = '🐍⚡'
            s4 = '大处着眼，小处着手。'
            return (",".join([s1, s2])._is_ascii,
                    "🐍⚡".join([s1, s2])._is_ascii,
                    ",".join([s1, s3])._is_ascii,
                    ",".join([s3, s4])._is_ascii)

        self.assertEqual(f(), (1, 0, 0, 0))

    def test_ascii_flag_getitem(self):
        @njit
        def f():
            s1 = 'abc123'
            s2 = '🐍⚡🐍⚡🐍⚡'
            return (s1[0]._is_ascii, s1[2:]._is_ascii, s2[0]._is_ascii,
                    s2[2:]._is_ascii)

        self.assertEqual(f(), (1, 1, 0, 0))

    def test_ascii_flag_add_mul(self):
        @njit
        def f():
            s1 = 'abc'
            s2 = '123'
            s3 = '🐍⚡'
            s4 = '大处着眼，小处着手。'
            return ((s1 + s2)._is_ascii,
                    (s1 + s3)._is_ascii,
                    (s3 + s4)._is_ascii,
                    (s1 * 2)._is_ascii,
                    (s3 * 2)._is_ascii)

        self.assertEqual(f(), (1, 0, 0, 1, 0))


class TestUnicodeIteration(BaseTest):

    def test_unicode_iter(self):
        pyfunc = iter_usecase
        cfunc = njit(pyfunc)
        for a in UNICODE_EXAMPLES:
            self.assertPreciseEqual(pyfunc(a), cfunc(a))

    def test_unicode_literal_iter(self):
        pyfunc = literal_iter_usecase
        cfunc = njit(pyfunc)
        self.assertPreciseEqual(pyfunc(), cfunc())

    def test_unicode_enumerate_iter(self):
        pyfunc = enumerated_iter_usecase
        cfunc = njit(pyfunc)
        for a in UNICODE_EXAMPLES:
            self.assertPreciseEqual(pyfunc(a), cfunc(a))

    def test_unicode_stopiteration_iter(self):
        self.disable_leak_check()
        pyfunc = iter_stopiteration_usecase
        cfunc = njit(pyfunc)
        for f in (pyfunc, cfunc):
            for a in UNICODE_EXAMPLES:
                with self.assertRaises(StopIteration):
                    f(a)

    def test_unicode_literal_stopiteration_iter(self):
        pyfunc = literal_iter_stopiteration_usecase
        cfunc = njit(pyfunc)
        for f in (pyfunc, cfunc):
            with self.assertRaises(StopIteration):
                f()


class TestUnicodeAuxillary(BaseTest):

    def test_ord(self):
        pyfunc = ord_usecase
        cfunc = njit(pyfunc)
        for ex in UNICODE_EXAMPLES:
            for a in ex:
                self.assertPreciseEqual(pyfunc(a), cfunc(a))

    def test_ord_invalid(self):
        self.disable_leak_check()

        pyfunc = ord_usecase
        cfunc = njit(pyfunc)

        # wrong number of chars
        for func in (pyfunc, cfunc):
            for ch in ('', 'abc'):
                with self.assertRaises(TypeError) as raises:
                    func(ch)
                self.assertIn('ord() expected a character',
                              str(raises.exception))

        # wrong type
        with self.assertRaises(TypingError) as raises:
            cfunc(1.23)
        self.assertIn(_header_lead, str(raises.exception))

    def test_chr(self):
        pyfunc = chr_usecase
        cfunc = njit(pyfunc)
        for ex in UNICODE_EXAMPLES:
            for x in ex:
                a = ord(x)
                self.assertPreciseEqual(pyfunc(a), cfunc(a))
        # test upper/lower bounds
        for a in (0x0, _MAX_UNICODE):
            self.assertPreciseEqual(pyfunc(a), cfunc(a))

    def test_chr_invalid(self):
        pyfunc = chr_usecase
        cfunc = njit(pyfunc)

        # value negative/>_MAX_UNICODE
        for func in (pyfunc, cfunc):
            for v in (-2, _MAX_UNICODE + 1):
                with self.assertRaises(ValueError) as raises:
                    func(v)
                self.assertIn("chr() arg not in range", str(raises.exception))

        # wrong type
        with self.assertRaises(TypingError) as raises:
            cfunc('abc')
        self.assertIn(_header_lead, str(raises.exception))

    def test_unicode_type_mro(self):
        # see issue #5635
        def bar(x):
            return True

        @overload(bar)
        def ol_bar(x):
            ok = False
            if isinstance(x, types.UnicodeType):
                if isinstance(x, types.Hashable):
                    ok = True
            return lambda x: ok

        @njit
        def foo(strinst):
            return bar(strinst)

        inst = "abc"
        self.assertEqual(foo.py_func(inst), foo(inst))
        self.assertIn(types.Hashable, types.unicode_type.__class__.__mro__)

    def test_f_strings(self):
        """test f-string support, which requires bytecode handling
        """
        # requires formatting (FORMAT_VALUE) and concatenation (BUILD_STRINGS)
        def impl1(a):
            return f"AA_{a+3}_B"

        # does not require concatenation
        def impl2(a):
            return f"{a+2}"

        # no expression
        def impl3(a):
            return f"ABC_{a}"

        # format spec not allowed
        def impl4(a):
            return f"ABC_{a:0}"

        # corner case: empty string
        def impl5():
            return f""  # noqa: F541

        self.assertEqual(impl1(3), njit(impl1)(3))
        self.assertEqual(impl2(2), njit(impl2)(2))
        # string input
        self.assertEqual(impl3("DE"), njit(impl3)("DE"))
        # check error when input type doesn't have str() implementation
        with self.assertRaises(TypingError) as raises:
            njit(impl3)(["A", "B"])
        msg = "No implementation of function Function(<class 'str'>)"
        self.assertIn(msg, str(raises.exception))
        # check error when format spec provided
        with self.assertRaises(UnsupportedError) as raises:
            njit(impl4)(["A", "B"])
        msg = "format spec in f-strings not supported yet"
        self.assertIn(msg, str(raises.exception))
        self.assertEqual(impl5(), njit(impl5)())


class IntTestCases(BaseTest):

    L = [
        ('0', 0),
        ('1', 1),
        ('9', 9),
        ('10', 10),
        ('99', 99),
        ('100', 100),
        ('314', 314),
        (repr(sys.maxsize), sys.maxsize),
    ]

    L_might_raise_error = [
        (' 314', 314),
        ('314 ', 314),
        ('  \t\t  314  \t\t  ', 314),
        ('  1x', ValueError),
        ('  1  ', 1),
        ('  1\02  ', ValueError),
        ('', ValueError),
        (' ', ValueError),
        ('  \t\t  ', ValueError),
        ("\u0200", ValueError)
    ]

    # https://github.com/python/cpython/blob/1960eb005e04b7ad8a91018088cfdb0646bc1ca0/Lib/test/test_grammar.py#L26    # noqa: E501
    VALID_UNDERSCORE_LITERALS = [
        '0_0_0',
        '4_2',
        '1_0000_0000',
        '0b1001_0100',
        '0xffff_ffff',
        '0o5_7_7',
        '1_00_00.5',
        '1_00_00.5e5',
        '1_00_00e5_1',
        '1e1_0',
        '.1_4',
        '.1_4e1',
        '0b_0',
        '0x_f',
        '0o_5',
        '1_00_00j',
        '1_00_00.5j',
        '1_00_00e5_1j',
        '.1_4j',
        '(1_2.5+3_3j)',
        '(.5_6j)',
    ]

    INVALID_UNDERSCORE_LITERALS = [
        # Trailing underscores:
        '0_',
        '42_',
        '1.4j_',
        '0x_',
        '0b1_',
        '0xf_',
        '0o5_',
        '0 if 1_Else 1',
        # Underscores in the base selector:
        '0_b0',
        '0_xf',
        '0_o5',
        # Old-style octal, still disallowed:
        # Python does allow the two below, don't know why they are put here.
        # '0_7',
        # '09_99',
        # Multiple consecutive underscores:
        '4_______2',
        '0.1__4',
        '0.1__4j',
        '0b1001__0100',
        '0xffff__ffff',
        '0x___',
        '0o5__77',
        '1e1__0',
        '1e1__0j',
        # Underscore right before a dot:
        '1_.4',
        '1_.4j',
        # Underscore right after a dot:
        '1._4',
        '1._4j',
        '._5',
        '._5j',
        # Underscore right after a sign:
        '1.0e+_1',
        '1.0e+_1j',
        # Underscore right before j:
        '1.4_j',
        '1.4e5_j',
        # Underscore right before e:
        '1_e1',
        '1.4_e1',
        '1.4_e1j',
        # Underscore right after e:
        '1e_1',
        '1.4e_1',
        '1.4e_1j',
        # Complex cases with parens:
        '(1+1.5_j_)',
        '(1+1.5_j)',
    ]

    # https://github.com/python/cpython/blob
    # /1960eb005e04b7ad8a91018088cfdb0646bc1ca0/Lib/test/test_int.py#L41    #
    # noqa: E501
    #NOTE: The test cases where conversion from non-str to int occur are removed.
    def test_basic(self):
        cfunc = njit(int_usecase)
        base_cfunc = njit(int_base_usecase)

        self.assertEqual(cfunc("-3"), -3)
        self.assertEqual(cfunc(" -3 "), -3)
        self.assertEqual(cfunc("\N{EM SPACE}-3\N{EN SPACE}"), -3)
        # Different base:
        self.assertEqual(base_cfunc("10", 16), 16)
        # Test conversion from strings and various anomalies
        for s, v in self.L:
            for sign in "", "+", "-":
                for prefix in "", " ", "\t", "  \t\t  ":
                    ss = prefix + sign + s
                    vv = v
                    if sign == "-" and v is not ValueError:
                        vv = -v
                    self.assertEqual(cfunc(ss), vv)

        # cases for border cases, -1<<63 and 1<<63-1
        self.assertEqual(cfunc(str(-1 - sys.maxsize)), -1 - sys.maxsize)
        self.assertEqual(cfunc(str(sys.maxsize)), sys.maxsize)

        self.assertEqual(base_cfunc('0o123', 0), 83)
        self.assertEqual(base_cfunc('0x123', 16), 291)

        # SF bug 1334662: int(string, base) wrong answers
        # Various representations of 2**32 evaluated to 0
        # rather than 2**32 in previous versions

        s = '100000000000000000000000000000000'
        self.assertEqual(base_cfunc(s, 2), 4294967296)
        self.assertEqual(base_cfunc('102002022201221111211', 3), 4294967296)
        self.assertEqual(base_cfunc('10000000000000000', 4), 4294967296)
        self.assertEqual(base_cfunc('32244002423141', 5), 4294967296)
        self.assertEqual(base_cfunc('1550104015504', 6), 4294967296)
        self.assertEqual(base_cfunc('211301422354', 7), 4294967296)
        self.assertEqual(base_cfunc('40000000000', 8), 4294967296)
        self.assertEqual(base_cfunc('12068657454', 9), 4294967296)
        self.assertEqual(base_cfunc('4294967296', 10), 4294967296)
        self.assertEqual(base_cfunc('1904440554', 11), 4294967296)
        self.assertEqual(base_cfunc('9ba461594', 12), 4294967296)
        self.assertEqual(base_cfunc('535a79889', 13), 4294967296)
        self.assertEqual(base_cfunc('2ca5b7464', 14), 4294967296)
        self.assertEqual(base_cfunc('1a20dcd81', 15), 4294967296)
        self.assertEqual(base_cfunc('100000000', 16), 4294967296)
        self.assertEqual(base_cfunc('a7ffda91', 17), 4294967296)
        self.assertEqual(base_cfunc('704he7g4', 18), 4294967296)
        self.assertEqual(base_cfunc('4f5aff66', 19), 4294967296)
        self.assertEqual(base_cfunc('3723ai4g', 20), 4294967296)
        self.assertEqual(base_cfunc('281d55i4', 21), 4294967296)
        self.assertEqual(base_cfunc('1fj8b184', 22), 4294967296)
        self.assertEqual(base_cfunc('1606k7ic', 23), 4294967296)
        self.assertEqual(base_cfunc('mb994ag', 24), 4294967296)
        self.assertEqual(base_cfunc('hek2mgl', 25), 4294967296)
        self.assertEqual(base_cfunc('dnchbnm', 26), 4294967296)
        self.assertEqual(base_cfunc('b28jpdm', 27), 4294967296)
        self.assertEqual(base_cfunc('8pfgih4', 28), 4294967296)
        self.assertEqual(base_cfunc('76beigg', 29), 4294967296)
        self.assertEqual(base_cfunc('5qmcpqg', 30), 4294967296)
        self.assertEqual(base_cfunc('4q0jto4', 31), 4294967296)
        self.assertEqual(base_cfunc('4000000', 32), 4294967296)
        self.assertEqual(base_cfunc('3aokq94', 33), 4294967296)
        self.assertEqual(base_cfunc('2qhxjli', 34), 4294967296)
        self.assertEqual(base_cfunc('2br45qb', 35), 4294967296)
        self.assertEqual(base_cfunc('1z141z4', 36), 4294967296)

        # tests with base 0
        # this fails on 3.0, but in 2.x the old octal syntax is allowed
        self.assertEqual(base_cfunc(' 0o123  ', 0), 83)
        self.assertEqual(base_cfunc(' 0o123  ', 0), 83)
        self.assertEqual(base_cfunc('000', 0), 0)
        self.assertEqual(base_cfunc('0o123', 0), 83)
        self.assertEqual(base_cfunc('0x123', 0), 291)
        self.assertEqual(base_cfunc('0b100', 0), 4)
        self.assertEqual(base_cfunc(' 0O123   ', 0), 83)
        self.assertEqual(base_cfunc(' 0X123  ', 0), 291)
        self.assertEqual(base_cfunc(' 0B100 ', 0), 4)

        # without base still base 10
        self.assertEqual(cfunc('0123'), 123)
        self.assertEqual(base_cfunc('0123', 10), 123)

        # tests with prefix and base != 0
        self.assertEqual(base_cfunc('0x123', 16), 291)
        self.assertEqual(base_cfunc('0o123', 8), 83)
        self.assertEqual(base_cfunc('0b100', 2), 4)
        self.assertEqual(base_cfunc('0X123', 16), 291)
        self.assertEqual(base_cfunc('0O123', 8), 83)
        self.assertEqual(base_cfunc('0B100', 2), 4)

    def test_basic_raise_error(self):
        self.disable_leak_check()

        cfunc = njit(int_usecase)
        base_cfunc = njit(int_base_usecase)

        # Test conversion from strings and various anomalies
        for s, v in self.L_might_raise_error:
            for sign in "", "+", "-":
                for prefix in "", " ", "\t", "  \t\t  ":
                    ss = prefix + sign + s
                    vv = v
                    if sign == "-" and v is not ValueError:
                        vv = -v
                    try:
                        self.assertEqual(cfunc(ss), vv)
                    except ValueError as e:
                        self.assertEqual(str(e), _INVALID_LITERAL_MSG)

        # Bug 1679: "0x" is not a valid hex literal
        self.assertRaises(ValueError, base_cfunc, "0x", 16)
        self.assertRaises(ValueError, base_cfunc, "0x", 0)

        self.assertRaises(ValueError, base_cfunc, "0o", 8)
        self.assertRaises(ValueError, base_cfunc, "0o", 0)

        self.assertRaises(ValueError, base_cfunc, "0b", 2)
        self.assertRaises(ValueError, base_cfunc, "0b", 0)

        # the code has special checks for the first character after the
        #  type prefix
        self.assertRaises(ValueError, base_cfunc, '0b2', 2)
        self.assertRaises(ValueError, base_cfunc, '0b02', 2)
        self.assertRaises(ValueError, base_cfunc, '0B2', 2)
        self.assertRaises(ValueError, base_cfunc, '0B02', 2)
        self.assertRaises(ValueError, base_cfunc, '0o8', 8)
        self.assertRaises(ValueError, base_cfunc, '0o08', 8)
        self.assertRaises(ValueError, base_cfunc, '0O8', 8)
        self.assertRaises(ValueError, base_cfunc, '0O08', 8)
        self.assertRaises(ValueError, base_cfunc, '0xg', 16)
        self.assertRaises(ValueError, base_cfunc, '0x0g', 16)
        self.assertRaises(ValueError, base_cfunc, '0Xg', 16)
        self.assertRaises(ValueError, base_cfunc, '0X0g', 16)

    def test_underscores(self):
        self.disable_leak_check()
        cfunc = njit(int_usecase)
        base_cfunc = njit(int_base_usecase)

        for lit in self.VALID_UNDERSCORE_LITERALS:
            if any(ch in lit for ch in '.eEjJ'):
                continue
            self.assertEqual(base_cfunc(lit, 0), eval(lit))
            lit_without_us = lit.replace('_', '')
            self.assertEqual(base_cfunc(lit, 0), base_cfunc(lit_without_us, 0))
        for lit in self.INVALID_UNDERSCORE_LITERALS:
            if any(ch in lit for ch in '.eEjJ'):
                continue
            self.assertRaises(ValueError, base_cfunc, lit, 0)
        # Additional test cases with bases != 0, only for the constructor:
        self.assertEqual(base_cfunc("1_00", 3), 9)
        self.assertEqual(cfunc("0_100"), 100)  # not valid as a literal!
        self.assertRaises(ValueError, cfunc, "_100")
        self.assertRaises(ValueError, cfunc, "+_100")
        self.assertRaises(ValueError, cfunc, "1__00")
        self.assertRaises(ValueError, cfunc, "100_")

    def test_int_base_limits(self):
        # Testing the supported limits of the int() base parameter.
        self.disable_leak_check()
        base_cfunc = njit(int_base_usecase)
        self.assertEqual(base_cfunc('0', 5), 0)
        with self.assertRaises(ValueError):
            base_cfunc('0', 1)
        with self.assertRaises(ValueError):
            base_cfunc('0', 37)
        with self.assertRaises(ValueError):
            base_cfunc('0', -909)  # An old magic value base from Python 2.
        # Bases 2 through 36 are supported.
        for base in range(2, 37):
            self.assertEqual(base_cfunc('0', base=base), 0)

    def test_int_base_bad_types(self):
        # Not integer types are not valid bases; issue16772
        self.disable_leak_check()
        base_cfunc = njit(int_base_usecase)
        with self.assertRaises(TypingError):
            base_cfunc('0', 5.5)
        with self.assertRaises(TypingError):
            base_cfunc('0', 5.0)

    def test_string_float(self):
        self.disable_leak_check()
        cfunc = njit(int_usecase)
        self.assertRaises(ValueError, cfunc, '1.2')

    def test_error_message(self):
        self.disable_leak_check()
        cfunc = njit(int_usecase)
        base_cfunc = njit(int_base_usecase)

        def check(s, base=None):
            with self.assertRaises(ValueError, msg=_INVALID_LITERAL_MSG) as cm:
                if base is None:
                    cfunc(s)
                else:
                    base_cfunc(s, base)
            self.assertEqual(cm.exception.args[0], _INVALID_LITERAL_MSG)

        check('\xbd')
        check('123\xbd')
        check('  123 456  ')

        check('123\x00')
        # SF bug 1545497: embedded NULs were not detected with explicit base
        check('123\x00', 10)
        check('123\x00 245', 20)
        check('123\x00 245', 16)
        check('123\x00245', 20)
        check('123\x00245', 16)
        # lone surrogate in Unicode string
        check('123\ud800')
        check('123\ud800', 10)

        # Add two assertions for unsupported types.
        unicode_regex = "The literal should be an UnicodeType, but we got a .*"
        with self.assertRaisesRegex(TypingError, unicode_regex):
            cfunc(("a", "b"))
        base_regex = "The base argument should be an Integer, but we got a .*"
        with self.assertRaisesRegex(TypingError, base_regex):
            base_cfunc('101', 0.5)

    def test_issue31619(self):
        base_cfunc = njit(int_base_usecase)
        s = '1_0_1_0_1_0_1_0_1_0_1_0_1_0_1_0_1_0_1_0_1_0_1_0_1_0_1_0_1_0_1'
        self.assertEqual(base_cfunc(s, 2), 0b1010101010101010101010101010101)
        self.assertEqual(base_cfunc('1_2_3_4_5_6_7_0_1_2_3', 8), 0o12345670123)
        self.assertEqual(base_cfunc('1_2_3_4_5_6_7_8_9', 16), 0x123456789)
        self.assertEqual(base_cfunc('1_2_3_4_5_6_7', 32), 1144132807)


if __name__ == '__main__':
    unittest.main()
