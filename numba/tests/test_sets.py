from __future__ import print_function

import numba.unittest_support as unittest

from collections import namedtuple
import contextlib
import itertools
import math
import random
import sys

import numpy as np

from numba.compiler import compile_isolated, Flags
from numba import jit, types
import numba.unittest_support as unittest
from .support import (TestCase, enable_pyobj_flags, MemoryLeakMixin, tag,
                      compile_function)


def _build_set_literal_usecase(code, args):
    code = code % {'initializer': ', '.join(repr(arg) for arg in args)}
    return compile_function('build_set', code, globals())

def set_literal_return_usecase(args):
    code = """if 1:
    def build_set():
        return {%(initializer)s}
    """
    return _build_set_literal_usecase(code, args)

def set_literal_convert_usecase(args):
    code = """if 1:
    def build_set():
        my_set = {%(initializer)s}
        return list(my_set)
    """
    return _build_set_literal_usecase(code, args)


def empty_constructor_usecase():
    s = set()
    s.add(1)
    return len(s)

def constructor_usecase(arg):
    s = set(arg)
    return len(s)

def iterator_usecase(arg):
    s = set(arg)
    l = []
    for v in s:
        l.append(v)
    return l

def update_usecase(a, b, c):
    s = set()
    s.update(a)
    s.update(b)
    s.update(c)
    return list(s)

def remove_usecase(a, b):
    s = set(a)
    for v in b:
        s.remove(v)
    return list(s)

def discard_usecase(a, b):
    s = set(a)
    for v in b:
        s.discard(v)
    return list(s)

def pop_usecase(a):
    s = set(a)
    l = []
    while len(s) > 0:
        l.append(s.pop())
    return l

def contains_usecase(a, b):
    s = set(a)
    l = []
    for v in b:
        l.append(v in s)
    return l

def difference_update_usecase(a, b):
    s = set(a)
    s.difference_update(set(b))
    return list(s)

def intersection_update_usecase(a, b):
    s = set(a)
    s.intersection_update(set(b))
    return list(s)

def symmetric_difference_update_usecase(a, b):
    s = set(a)
    s.symmetric_difference_update(set(b))
    return list(s)

def isdisjoint_usecase(a, b):
    return set(a).isdisjoint(set(b))

def issubset_usecase(a, b):
    return set(a).issubset(set(b))

def issuperset_usecase(a, b):
    return set(a).issuperset(set(b))

def clear_usecase(a):
    s = set(a)
    s.clear()
    return len(s), list(s)

def copy_usecase(a):
    s = set(a)
    ss = s.copy()
    s.pop()
    return len(ss), list(ss)

def copy_usecase_empty(a):
    s = set(a)
    s.clear()
    ss = s.copy()
    s.add(42)
    return len(ss), list(ss)

def copy_usecase_deleted(a, b):
    s = set(a)
    s.remove(b)
    ss = s.copy()
    s.pop()
    return len(ss), list(ss)

def difference_usecase(a, b):
    sa = set(a)
    s = sa.difference(set(b))
    return list(s)

def intersection_usecase(a, b):
    sa = set(a)
    s = sa.intersection(set(b))
    return list(s)

def symmetric_difference_usecase(a, b):
    sa = set(a)
    s = sa.symmetric_difference(set(b))
    return list(s)

def union_usecase(a, b):
    sa = set(a)
    s = sa.union(set(b))
    return list(s)


def make_operator_usecase(op):
    code = """if 1:
    def operator_usecase(a, b):
        s = set(a) %(op)s set(b)
        return list(s)
    """ % dict(op=op)
    return compile_function('operator_usecase', code, globals())

def make_comparison_usecase(op):
    code = """if 1:
    def comparison_usecase(a, b):
        return set(a) %(op)s set(b)
    """ % dict(op=op)
    return compile_function('comparison_usecase', code, globals())


def unique_usecase(src):
    seen = set()
    res = []
    for v in src:
        if v not in seen:
            seen.add(v)
            res.append(v)
    return res


needs_set_literals = unittest.skipIf(sys.version_info < (2, 7),
                                     "set literals unavailable before Python 2.7")


class BaseTest(MemoryLeakMixin, TestCase):

    def setUp(self):
        super(BaseTest, self).setUp()
        self.rnd = random.Random(42)

    def _range(self, stop):
        return np.arange(int(stop))

    def _random_choice(self, seq, n):
        """
        Choose *n* possibly duplicate items from sequence.
        """
        # np.random.choice() doesn't exist on Numpy 1.6
        l = [self.rnd.choice(list(seq)) for i in range(n)]
        if isinstance(seq, np.ndarray):
            return np.array(l, dtype=seq.dtype)
        else:
            return l

    def duplicates_array(self, n):
        """
        Get a 1d array with many duplicate values.
        """
        a = self._range(np.sqrt(n))
        return self._random_choice(a, n)

    def sparse_array(self, n):
        """
        Get a 1d array with values spread around.
        """
        # Note two calls to sparse_array() should generate reasonable overlap
        a = self._range(n ** 1.3)
        return self._random_choice(a, n)

    def _assert_equal_unordered(self, a, b):
        if isinstance(a, tuple):
            self.assertIsInstance(b, tuple)
            for u, v in zip(a, b):
                self._assert_equal_unordered(u, v)
        elif isinstance(a, list):
            self.assertIsInstance(b, list)
            self.assertPreciseEqual(sorted(a), sorted(b))
        else:
            self.assertPreciseEqual(a, b)

    def unordered_checker(self, pyfunc):
        cfunc = jit(nopython=True)(pyfunc)
        def check(*args):
            expected = pyfunc(*args)
            got = cfunc(*args)
            self._assert_equal_unordered(expected, got)
        return check


class TestSetLiterals(BaseTest):

    @needs_set_literals
    def test_build_set(self, flags=enable_pyobj_flags):
        pyfunc = set_literal_return_usecase((1, 2, 3, 2))
        self.run_nullary_func(pyfunc, flags=flags)

    @needs_set_literals
    def test_build_heterogenous_set(self, flags=enable_pyobj_flags):
        pyfunc = set_literal_return_usecase((1, 2.0, 3j, 2))
        self.run_nullary_func(pyfunc, flags=flags)
        # Check that items are inserted in the right order (here the
        # result will be {2}, not {2.0})
        pyfunc = set_literal_return_usecase((2.0, 2))
        got, expected = self.run_nullary_func(pyfunc, flags=flags)
        self.assertIs(type(got.pop()), type(expected.pop()))

    @tag('important')
    @needs_set_literals
    def test_build_set_nopython(self):
        arg = list(self.sparse_array(50))
        pyfunc = set_literal_convert_usecase(arg)
        cfunc = jit(nopython=True)(pyfunc)

        expected = pyfunc()
        got = cfunc()
        self.assertPreciseEqual(sorted(expected), sorted(got))


class TestSets(BaseTest):

    def test_constructor(self):
        pyfunc = empty_constructor_usecase
        cfunc = jit(nopython=True)(pyfunc)
        self.assertPreciseEqual(cfunc(), pyfunc())

        pyfunc = constructor_usecase
        cfunc = jit(nopython=True)(pyfunc)
        def check(arg):
            self.assertPreciseEqual(pyfunc(arg), cfunc(arg))

        check((1, 2, 3, 2, 7))
        check(self.duplicates_array(200))
        check(self.sparse_array(200))

    @tag('important')
    def test_iterator(self):
        pyfunc = iterator_usecase
        check = self.unordered_checker(pyfunc)

        check((1, 2, 3, 2, 7))
        check(self.duplicates_array(200))
        check(self.sparse_array(200))

    @tag('important')
    def test_update(self):
        pyfunc = update_usecase
        check = self.unordered_checker(pyfunc)

        a, b, c = (1, 2, 4, 9), (2, 3, 5, 11, 42), (4, 5, 6, 42)
        check(a, b, c)

        a = self.sparse_array(50)
        b = self.duplicates_array(50)
        c = self.sparse_array(50)
        check(a, b, c)

    def test_remove(self):
        pyfunc = remove_usecase
        check = self.unordered_checker(pyfunc)

        a = (1, 2, 3, 5, 8, 42)
        b = (5, 2, 8)
        check(a, b)

    def test_remove_error(self):
        # References are leaked on exception
        self.disable_leak_check()

        pyfunc = remove_usecase
        cfunc = jit(nopython=True)(pyfunc)
        with self.assertRaises(KeyError) as raises:
            cfunc((1, 2, 3), (5, ))

    @tag('important')
    def test_discard(self):
        pyfunc = discard_usecase
        check = self.unordered_checker(pyfunc)

        a = (1, 2, 3, 5, 8, 42)
        b = (5, 2, 8)
        check(a, b)
        a = self.sparse_array(50)
        b = self.sparse_array(50)
        check(a, b)

    @tag('important')
    def test_pop(self):
        pyfunc = pop_usecase
        check = self.unordered_checker(pyfunc)

        check((2, 3, 55, 11, 8, 42))
        check(self.sparse_array(50))

    @tag('important')
    def test_contains(self):
        pyfunc = contains_usecase
        cfunc = jit(nopython=True)(pyfunc)
        def check(a, b):
            self.assertPreciseEqual(pyfunc(a, b), cfunc(a, b))

        a = (1, 2, 3, 5, 42)
        b = (5, 2, 8, 3)
        check(a, b)

    def _test_xxx_update(self, pyfunc):
        check = self.unordered_checker(pyfunc)

        a, b = (1, 2, 4, 11), (2, 3, 5, 11, 42)
        check(a, b)

        sizes = (0, 50, 500)
        for na, nb in itertools.product(sizes, sizes):
            a = self.sparse_array(na)
            b = self.sparse_array(nb)
            check(a, b)

    def test_difference_update(self):
        self._test_xxx_update(difference_update_usecase)

    def test_intersection_update(self):
        self._test_xxx_update(intersection_update_usecase)

    def test_symmetric_difference_update(self):
        self._test_xxx_update(symmetric_difference_update_usecase)

    def _test_comparator(self, pyfunc):
        cfunc = jit(nopython=True)(pyfunc)
        def check(a, b):
            self.assertPreciseEqual(pyfunc(a, b), cfunc(a, b))

        a, b = map(set, [(1, 2, 4, 11), (2, 3, 5, 11, 42)])
        args = [a & b, a - b, a | b, a ^ b]
        args = [tuple(x) for x in args]
        for a, b in itertools.product(args, args):
            check(a, b)

    def test_isdisjoint(self):
        self._test_comparator(isdisjoint_usecase)

    def test_issubset(self):
        self._test_comparator(issubset_usecase)

    def test_issuperset(self):
        self._test_comparator(issuperset_usecase)

    def test_clear(self):
        pyfunc = clear_usecase
        check = self.unordered_checker(pyfunc)

        check((1, 2, 4, 11))
        check(self.sparse_array(50))

    def test_copy(self):
        # Source set doesn't have any deleted entries
        pyfunc = copy_usecase
        check = self.unordered_checker(pyfunc)
        check((1, 2, 4, 11))
        check(self.sparse_array(50))

        pyfunc = copy_usecase_empty
        check = self.unordered_checker(pyfunc)
        check((1,))

        # Source set has deleted entries
        pyfunc = copy_usecase_deleted
        check = self.unordered_checker(pyfunc)
        check((1, 2, 4, 11), 2)
        a = self.sparse_array(50)
        check(a, a[len(a) // 2])

    def _test_set_operator(self, pyfunc):
        check = self.unordered_checker(pyfunc)

        a, b = (1, 2, 4, 11), (2, 3, 5, 11, 42)
        check(a, b)

        sizes = (0, 50, 500)
        for na, nb in itertools.product(sizes, sizes):
            a = self.sparse_array(na)
            b = self.sparse_array(nb)
            check(a, b)

    def test_difference(self):
        self._test_set_operator(difference_usecase)

    def test_intersection(self):
        self._test_set_operator(intersection_usecase)

    def test_symmetric_difference(self):
        self._test_set_operator(symmetric_difference_usecase)

    def test_union(self):
        self._test_set_operator(union_usecase)

    def test_and(self):
        self._test_set_operator(make_operator_usecase('&'))

    def test_or(self):
        self._test_set_operator(make_operator_usecase('|'))

    def test_sub(self):
        self._test_set_operator(make_operator_usecase('-'))

    def test_xor(self):
        self._test_set_operator(make_operator_usecase('^'))

    def test_eq(self):
        self._test_set_operator(make_comparison_usecase('=='))

    def test_ne(self):
        self._test_set_operator(make_comparison_usecase('!='))

    def test_le(self):
        self._test_set_operator(make_comparison_usecase('<='))

    def test_lt(self):
        self._test_set_operator(make_comparison_usecase('<'))

    def test_ge(self):
        self._test_set_operator(make_comparison_usecase('>='))

    def test_gt(self):
        self._test_set_operator(make_comparison_usecase('>'))


class OtherTypesTest(object):

    def test_constructor(self):
        pyfunc = empty_constructor_usecase
        cfunc = jit(nopython=True)(pyfunc)
        self.assertPreciseEqual(cfunc(), pyfunc())

        pyfunc = constructor_usecase
        cfunc = jit(nopython=True)(pyfunc)
        def check(arg):
            self.assertPreciseEqual(pyfunc(arg), cfunc(arg))

        check(self.duplicates_array(200))
        check(self.sparse_array(200))

    def test_iterator(self):
        pyfunc = iterator_usecase
        check = self.unordered_checker(pyfunc)

        check(self.duplicates_array(200))
        check(self.sparse_array(200))

    @tag('important')
    def test_update(self):
        pyfunc = update_usecase
        check = self.unordered_checker(pyfunc)

        a = self.sparse_array(50)
        b = self.duplicates_array(50)
        c = self.sparse_array(50)
        check(a, b, c)


class TestFloatSets(OtherTypesTest, BaseTest):
    """
    Test sets with floating-point keys.
    """
    # Only a few basic tests here, as the sanity of most operations doesn't
    # depend on the key type.

    def _range(self, stop):
        return np.arange(stop, dtype=np.float32) * np.float32(0.1)


class TestTupleSets(OtherTypesTest, BaseTest):
    """
    Test sets with tuple keys.
    """
    def _range(self, stop):
        a = np.arange(stop, dtype=np.int64)
        b = a & 0x5555555555555555
        c = (a & 0xaaaaaaaa).astype(np.int32)
        d = ((a >> 32) & 1).astype(np.bool_)
        return list(zip(b, c, d))


class TestExamples(BaseTest):
    """
    Examples of using sets.
    """

    @tag('important')
    def test_unique(self):
        pyfunc = unique_usecase
        check = self.unordered_checker(pyfunc)

        check(self.duplicates_array(200))
        check(self.sparse_array(200))


if __name__ == '__main__':
    unittest.main()
