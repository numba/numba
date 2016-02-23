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
from .support import TestCase, enable_pyobj_flags, nrt_flags, MemoryLeakMixin, tag


def _build_set_literal_usecase(code, args):
    ns = {}
    src = code % {'initializer': ', '.join(repr(arg) for arg in args)}
    code = compile(src, '<>', 'exec')
    eval(code, ns)
    return ns['build_set']

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


needs_set_literals = unittest.skipIf(sys.version_info < (2, 7),
                                     "set literals unavailable before Python 2.7")


class BaseTest(MemoryLeakMixin, TestCase):

    def setUp(self):
        super(BaseTest, self).setUp()
        self.rnd = random.Random(42)

    def _random_choice(self, seq, n):
        """
        Choose *n* possibly duplicate items from sequence.
        """
        # np.random.choice() doesn't exist on Numpy 1.6
        if isinstance(seq, np.ndarray):
            seq = list(seq)
        return np.array([self.rnd.choice(seq) for i in range(n)])

    def duplicates_array(self, n):
        """
        Get a 1d array with many duplicate values.
        """
        a = np.arange(int(np.sqrt(n)))
        return self._random_choice(a, n)

    def sparse_array(self, n):
        """
        Get a 1d array with values spread around.
        """
        # Note two calls to sparse_array() should generate reasonable overlap
        a = np.arange(int(n ** 1.3))
        return self._random_choice(a, n)

    def unordered_checker(self, pyfunc):
        cfunc = jit(nopython=True)(pyfunc)
        def check(*args):
            self.assertPreciseEqual(sorted(pyfunc(*args)),
                                    sorted(cfunc(*args)))
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

    def test_iterator(self):
        pyfunc = iterator_usecase
        check = self.unordered_checker(pyfunc)

        check((1, 2, 3, 2, 7))
        check(self.duplicates_array(200))
        check(self.sparse_array(200))

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

    def test_discard(self):
        pyfunc = discard_usecase
        check = self.unordered_checker(pyfunc)

        a = (1, 2, 3, 5, 8, 42)
        b = (5, 2, 8)
        check(a, b)
        a = self.sparse_array(50)
        b = self.sparse_array(50)
        check(a, b)

    def test_pop(self):
        pyfunc = pop_usecase
        check = self.unordered_checker(pyfunc)

        check((2, 3, 55, 11, 8, 42))
        check(self.sparse_array(50))

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

        sizes = (50, 500)
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


if __name__ == '__main__':
    unittest.main()
