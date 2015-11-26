from __future__ import print_function, division, absolute_import

import collections
import itertools

import numpy as np

from numba import unittest_support as unittest
from numba.compiler import compile_isolated
from numba import jit, types
from .support import TestCase, MemoryLeakMixin


Rect = collections.namedtuple('Rect', ('width', 'height'))

Point = collections.namedtuple('Point', ('x', 'y', 'z'))

Empty = collections.namedtuple('Empty', ())

def tuple_return_usecase(a, b):
    return a, b

def tuple_first(tup):
    a, b = tup
    return a

def tuple_second(tup):
    a, b = tup
    return b

def tuple_index(tup, idx):
    return tup[idx]

def len_usecase(tup):
    return len(tup)

def eq_usecase(a, b):
    return a == b

def ne_usecase(a, b):
    return a != b

def gt_usecase(a, b):
    return a > b

def ge_usecase(a, b):
    return a >= b

def lt_usecase(a, b):
    return a < b

def le_usecase(a, b):
    return a <= b

def bool_usecase(tup):
    return bool(tup), (3 if tup else 2)

def getattr_usecase(tup):
    return tup.z, tup.y, tup.x

def make_point(a, b, c):
    return Point(a, b, c)

def make_point_kws(a, b, c):
    return Point(z=c, y=b, x=a)

def make_point_nrt(n):
    r = Rect(list(range(n)), np.zeros(n + 1))
    # This also exercises attribute access
    p = Point(r, len(r.width), len(r.height))
    return p

def type_usecase(tup, *args):
    return type(tup)(*args)


class TestTupleReturn(TestCase):

    def test_array_tuple(self):
        aryty = types.Array(types.float64, 1, 'C')
        cres = compile_isolated(tuple_return_usecase, (aryty, aryty))
        a = b = np.arange(5, dtype='float64')
        ra, rb = cres.entry_point(a, b)
        self.assertTrue((ra == a).all())
        self.assertTrue((rb == b).all())
        del a, b
        self.assertTrue((ra == rb).all())

    def test_scalar_tuple(self):
        scalarty = types.float32
        cres = compile_isolated(tuple_return_usecase, (scalarty, scalarty))
        a = b = 1
        ra, rb = cres.entry_point(a, b)
        self.assertEqual(ra, a)
        self.assertEqual(rb, b)

    def test_hetero_tuple(self):
        alltypes = []
        allvalues = []

        alltypes.append((types.int32, types.int64))
        allvalues.append((1, 2))

        alltypes.append((types.float32, types.float64))
        allvalues.append((1.125, .25))

        alltypes.append((types.int32, types.float64))
        allvalues.append((1231, .5))

        for (ta, tb), (a, b) in zip(alltypes, allvalues):
            cres = compile_isolated(tuple_return_usecase, (ta, tb))
            ra, rb = cres.entry_point(a, b)
            self.assertPreciseEqual((ra, rb), (a, b))


class TestTuplePassing(TestCase):

    def test_unituple(self):
        tuple_type = types.UniTuple(types.int32, 2)
        cr_first = compile_isolated(tuple_first, (tuple_type,))
        cr_second = compile_isolated(tuple_second, (tuple_type,))
        self.assertPreciseEqual(cr_first.entry_point((4, 5)), 4)
        self.assertPreciseEqual(cr_second.entry_point((4, 5)), 5)

    def test_hetero_tuple(self):
        tuple_type = types.Tuple((types.int64, types.float32))
        cr_first = compile_isolated(tuple_first, (tuple_type,))
        cr_second = compile_isolated(tuple_second, (tuple_type,))
        self.assertPreciseEqual(cr_first.entry_point((2**61, 1.5)), 2**61)
        self.assertPreciseEqual(cr_second.entry_point((2**61, 1.5)), 1.5)


class TestOperations(TestCase):

    def test_len(self):
        pyfunc = len_usecase
        cr = compile_isolated(pyfunc,
                              [types.Tuple((types.int64, types.float32))])
        self.assertPreciseEqual(cr.entry_point((4, 5)), 2)
        cr = compile_isolated(pyfunc,
                              [types.UniTuple(types.int64, 3)])
        self.assertPreciseEqual(cr.entry_point((4, 5, 6)), 3)

    def test_index(self):
        pyfunc = tuple_index
        cr = compile_isolated(pyfunc,
                              [types.UniTuple(types.int64, 3), types.int64])
        tup = (4, 5, 6)
        for i in range(len(tup)):
            self.assertPreciseEqual(cr.entry_point(tup, i), tup[i])

    def test_bool(self):
        pyfunc = bool_usecase
        cr = compile_isolated(pyfunc,
                              [types.Tuple((types.int64, types.int32))])
        args = ((4, 5),)
        self.assertPreciseEqual(cr.entry_point(*args), pyfunc(*args))
        cr = compile_isolated(pyfunc,
                              [types.UniTuple(types.int64, 3)])
        args = ((4, 5, 6),)
        self.assertPreciseEqual(cr.entry_point(*args), pyfunc(*args))
        cr = compile_isolated(pyfunc,
                              [types.Tuple(())])
        self.assertPreciseEqual(cr.entry_point(()), pyfunc(()))

    def _test_compare(self, pyfunc):
        def eq(pyfunc, cfunc, args):
            self.assertIs(cfunc(*args), pyfunc(*args),
                          "mismatch for arguments %s" % (args,))

        # Same-sized tuples
        argtypes = [types.Tuple((types.int64, types.float32)),
                    types.UniTuple(types.int32, 2)]
        for ta, tb in itertools.product(argtypes, argtypes):
            cr = compile_isolated(pyfunc, (ta, tb))
            cfunc = cr.entry_point
            for args in [((4, 5), (4, 5)),
                         ((4, 5), (4, 6)),
                         ((4, 6), (4, 5)),
                         ((4, 5), (5, 4))]:
                eq(pyfunc, cfunc, args)
        # Different-sized tuples
        argtypes = [types.Tuple((types.int64, types.float32)),
                    types.UniTuple(types.int32, 3)]
        cr = compile_isolated(pyfunc, tuple(argtypes))
        cfunc = cr.entry_point
        for args in [((4, 5), (4, 5, 6)),
                     ((4, 5), (4, 4, 6)),
                     ((4, 5), (4, 6, 7))]:
            eq(pyfunc, cfunc, args)

    def test_eq(self):
        self._test_compare(eq_usecase)

    def test_ne(self):
        self._test_compare(ne_usecase)

    def test_gt(self):
        self._test_compare(gt_usecase)

    def test_ge(self):
        self._test_compare(ge_usecase)

    def test_lt(self):
        self._test_compare(lt_usecase)

    def test_le(self):
        self._test_compare(le_usecase)


class TestNamedTuple(TestCase, MemoryLeakMixin):

    def test_unpack(self):
        def check(p):
            for pyfunc in tuple_first, tuple_second:
                cfunc = jit(nopython=True)(pyfunc)
                self.assertPreciseEqual(cfunc(p), pyfunc(p))

        # Homogenous
        check(Rect(4, 5))
        # Heterogenous
        check(Rect(4, 5.5))

    def test_len(self):
        def check(p):
            pyfunc = len_usecase
            cfunc = jit(nopython=True)(pyfunc)
            self.assertPreciseEqual(cfunc(p), pyfunc(p))

        # Homogenous
        check(Rect(4, 5))
        check(Point(4, 5, 6))
        # Heterogenous
        check(Rect(4, 5.5))
        check(Point(4, 5.5, 6j))

    def test_index(self):
        pyfunc = tuple_index
        cfunc = jit(nopython=True)(pyfunc)

        p = Point(4, 5, 6)
        for i in range(len(p)):
            self.assertPreciseEqual(cfunc(p, i), pyfunc(p, i))

    def test_bool(self):
        def check(p):
            pyfunc = bool_usecase
            cfunc = jit(nopython=True)(pyfunc)
            self.assertPreciseEqual(cfunc(p), pyfunc(p))

        # Homogenous
        check(Rect(4, 5))
        # Heterogenous
        check(Rect(4, 5.5))
        check(Empty())

    def _test_compare(self, pyfunc):
        def eq(pyfunc, cfunc, args):
            self.assertIs(cfunc(*args), pyfunc(*args),
                          "mismatch for arguments %s" % (args,))

        cfunc = jit(nopython=True)(pyfunc)

        # Same-sized named tuples
        for a, b in [((4, 5), (4, 5)),
                     ((4, 5), (4, 6)),
                     ((4, 6), (4, 5)),
                     ((4, 5), (5, 4))]:
            eq(pyfunc, cfunc, (Rect(*a), Rect(*b)))

        # Different-sized named tuples
        for a, b in [((4, 5), (4, 5, 6)),
                     ((4, 5), (4, 4, 6)),
                     ((4, 5), (4, 6, 7))]:
            eq(pyfunc, cfunc, (Rect(*a), Point(*b)))

    def test_eq(self):
        self._test_compare(eq_usecase)

    def test_ne(self):
        self._test_compare(ne_usecase)

    def test_gt(self):
        self._test_compare(gt_usecase)

    def test_ge(self):
        self._test_compare(ge_usecase)

    def test_lt(self):
        self._test_compare(lt_usecase)

    def test_le(self):
        self._test_compare(le_usecase)

    def test_getattr(self):
        pyfunc = getattr_usecase
        cfunc = jit(nopython=True)(pyfunc)

        for args in (4, 5, 6), (4, 5.5, 6j):
            p = Point(*args)
            self.assertPreciseEqual(cfunc(p), pyfunc(p))

    def test_construct(self):
        def check(pyfunc):
            cfunc = jit(nopython=True)(pyfunc)
            for args in (4, 5, 6), (4, 5.5, 6j):
                expected = pyfunc(*args)
                got = cfunc(*args)
                self.assertIs(type(got), type(expected))
                self.assertPreciseEqual(got, expected)

        check(make_point)
        check(make_point_kws)

    def test_type(self):
        # Test the type() built-in on named tuples
        pyfunc = type_usecase
        cfunc = jit(nopython=True)(pyfunc)

        arg_tuples = [(4, 5, 6), (4, 5.5, 6j)]
        for tup_args, args in itertools.product(arg_tuples, arg_tuples):
            tup = Point(*tup_args)
            expected = pyfunc(tup, *args)
            got = cfunc(tup, *args)
            self.assertIs(type(got), type(expected))
            self.assertPreciseEqual(got, expected)


class TestNamedTupleNRT(TestCase, MemoryLeakMixin):

    def test_return(self):
        # Check returning a namedtuple with a list inside it
        pyfunc = make_point_nrt
        cfunc = jit(nopython=True)(pyfunc)

        for arg in (3, 0):
            expected = pyfunc(arg)
            got = cfunc(arg)
            self.assertIs(type(got), type(expected))
            self.assertPreciseEqual(got, expected)


if __name__ == '__main__':
    unittest.main()
