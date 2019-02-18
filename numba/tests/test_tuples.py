from __future__ import print_function, division, absolute_import

import collections
import itertools

import numpy as np

from numba import unittest_support as unittest
from numba.compiler import compile_isolated
from numba import jit, types, errors, utils
from .support import TestCase, MemoryLeakMixin, tag


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

def tuple_index_static(tup):
    # Note the negative index
    return tup[-2]

def tuple_slice2(tup):
    return tup[1:-1]

def tuple_slice3(tup):
    return tup[1::2]

def len_usecase(tup):
    return len(tup)

def add_usecase(a, b):
    return a + b

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

def in_usecase(a, b):
    return a in b

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

def identity(tup):
    return tup

def index_method_usecase(tup, value):
    return tup.index(value)


class TestTupleReturn(TestCase):

    @tag('important')
    def test_array_tuple(self):
        aryty = types.Array(types.float64, 1, 'C')
        cres = compile_isolated(tuple_return_usecase, (aryty, aryty))
        a = b = np.arange(5, dtype='float64')
        ra, rb = cres.entry_point(a, b)
        self.assertPreciseEqual(ra, a)
        self.assertPreciseEqual(rb, b)
        del a, b
        self.assertPreciseEqual(ra, rb)

    def test_scalar_tuple(self):
        scalarty = types.float32
        cres = compile_isolated(tuple_return_usecase, (scalarty, scalarty))
        a = b = 1
        ra, rb = cres.entry_point(a, b)
        self.assertEqual(ra, a)
        self.assertEqual(rb, b)

    @tag('important')
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

    @tag('important')
    def test_unituple(self):
        tuple_type = types.UniTuple(types.int32, 2)
        cr_first = compile_isolated(tuple_first, (tuple_type,))
        cr_second = compile_isolated(tuple_second, (tuple_type,))
        self.assertPreciseEqual(cr_first.entry_point((4, 5)), 4)
        self.assertPreciseEqual(cr_second.entry_point((4, 5)), 5)

    @tag('important')
    def test_hetero_tuple(self):
        tuple_type = types.Tuple((types.int64, types.float32))
        cr_first = compile_isolated(tuple_first, (tuple_type,))
        cr_second = compile_isolated(tuple_second, (tuple_type,))
        self.assertPreciseEqual(cr_first.entry_point((2**61, 1.5)), 2**61)
        self.assertPreciseEqual(cr_second.entry_point((2**61, 1.5)), 1.5)

    def test_size_mismatch(self):
        # Issue #1638: tuple size should be checked when unboxing
        tuple_type = types.UniTuple(types.int32, 2)
        cr = compile_isolated(tuple_first, (tuple_type,))
        with self.assertRaises(ValueError) as raises:
            cr.entry_point((4, 5, 6))
        self.assertEqual(str(raises.exception),
                         "size mismatch for tuple, expected 2 element(s) but got 3")


class TestOperations(TestCase):

    @tag('important')
    def test_len(self):
        pyfunc = len_usecase
        cr = compile_isolated(pyfunc,
                              [types.Tuple((types.int64, types.float32))])
        self.assertPreciseEqual(cr.entry_point((4, 5)), 2)
        cr = compile_isolated(pyfunc,
                              [types.UniTuple(types.int64, 3)])
        self.assertPreciseEqual(cr.entry_point((4, 5, 6)), 3)

    @tag('important')
    def test_index(self):
        pyfunc = tuple_index
        cr = compile_isolated(pyfunc,
                              [types.UniTuple(types.int64, 3), types.int64])
        tup = (4, 3, 6)
        for i in range(len(tup)):
            self.assertPreciseEqual(cr.entry_point(tup, i), tup[i])

        # test negative indexing
        for i in range(len(tup) + 1):
            self.assertPreciseEqual(cr.entry_point(tup, -i), tup[-i])

        # oob indexes, +ve then -ve
        with self.assertRaises(IndexError) as raises:
            cr.entry_point(tup, len(tup))
        self.assertEqual("tuple index out of range", str(raises.exception))
        with self.assertRaises(IndexError) as raises:
            cr.entry_point(tup, -(len(tup) + 1))
        self.assertEqual("tuple index out of range", str(raises.exception))

        # Test empty tuple
        cr = compile_isolated(pyfunc,
                              [types.UniTuple(types.int64, 0), types.int64])
        with self.assertRaises(IndexError) as raises:
            cr.entry_point((), 0)
        self.assertEqual("tuple index out of range", str(raises.exception))

        # test uintp indexing (because, e.g., parfor generates unsigned prange)
        cr = compile_isolated(pyfunc,
                              [types.UniTuple(types.int64, 3), types.uintp])
        for i in range(len(tup)):
            self.assertPreciseEqual(cr.entry_point(tup, types.uintp(i)), tup[i])

        # With a compile-time static index (the code generation path is different)
        pyfunc = tuple_index_static
        for typ in (types.UniTuple(types.int64, 4),
                    types.Tuple((types.int64, types.int32, types.int64, types.int32))):
            cr = compile_isolated(pyfunc, (typ,))
            tup = (4, 3, 42, 6)
            self.assertPreciseEqual(cr.entry_point(tup), pyfunc(tup))

        typ = types.UniTuple(types.int64, 1)
        with self.assertTypingError():
            cr = compile_isolated(pyfunc, (typ,))

    def test_in(self):
        pyfunc = in_usecase
        cr = compile_isolated(pyfunc,
                              [types.int64, types.UniTuple(types.int64, 3)])
        tup = (4, 1, 5)
        for i in range(5):
            self.assertPreciseEqual(cr.entry_point(i, tup), pyfunc(i, tup))

    def check_slice(self, pyfunc):
        tup = (4, 5, 6, 7)
        cr = compile_isolated(pyfunc,
                              [types.UniTuple(types.int64, 4)])
        self.assertPreciseEqual(cr.entry_point(tup), pyfunc(tup))
        cr = compile_isolated(
            pyfunc,
            [types.Tuple((types.int64, types.int32, types.int64, types.int32))])
        self.assertPreciseEqual(cr.entry_point(tup), pyfunc(tup))

    def test_slice2(self):
        self.check_slice(tuple_slice2)

    def test_slice3(self):
        self.check_slice(tuple_slice3)

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

    @tag('important')
    def test_add(self):
        pyfunc = add_usecase
        samples = [(types.Tuple(()), ()),
                   (types.UniTuple(types.int32, 0), ()),
                   (types.UniTuple(types.int32, 1), (42,)),
                   (types.Tuple((types.int64, types.float32)), (3, 4.5)),
                   ]
        for (ta, a), (tb, b) in itertools.product(samples, samples):
            cr = compile_isolated(pyfunc, (ta, tb))
            expected = pyfunc(a, b)
            got = cr.entry_point(a, b)
            self.assertPreciseEqual(got, expected, msg=(ta, tb))

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

    @tag('important')
    def test_eq(self):
        self._test_compare(eq_usecase)

    @tag('important')
    def test_ne(self):
        self._test_compare(ne_usecase)

    @tag('important')
    def test_gt(self):
        self._test_compare(gt_usecase)

    @tag('important')
    def test_ge(self):
        self._test_compare(ge_usecase)

    @tag('important')
    def test_lt(self):
        self._test_compare(lt_usecase)

    @tag('important')
    def test_le(self):
        self._test_compare(le_usecase)


class TestNamedTuple(TestCase, MemoryLeakMixin):

    def test_unpack(self):
        def check(p):
            for pyfunc in tuple_first, tuple_second:
                cfunc = jit(nopython=True)(pyfunc)
                self.assertPreciseEqual(cfunc(p), pyfunc(p))

        # Homogeneous
        check(Rect(4, 5))
        # Heterogeneous
        check(Rect(4, 5.5))

    def test_len(self):
        def check(p):
            pyfunc = len_usecase
            cfunc = jit(nopython=True)(pyfunc)
            self.assertPreciseEqual(cfunc(p), pyfunc(p))

        # Homogeneous
        check(Rect(4, 5))
        check(Point(4, 5, 6))
        # Heterogeneous
        check(Rect(4, 5.5))
        check(Point(4, 5.5, 6j))

    def test_index(self):
        pyfunc = tuple_index
        cfunc = jit(nopython=True)(pyfunc)

        p = Point(4, 5, 6)
        for i in range(len(p)):
            self.assertPreciseEqual(cfunc(p, i), pyfunc(p, i))

        # test uintp indexing (because, e.g., parfor generates unsigned prange)
        for i in range(len(p)):
            self.assertPreciseEqual(cfunc(p, types.uintp(i)), pyfunc(p, i))

    def test_bool(self):
        def check(p):
            pyfunc = bool_usecase
            cfunc = jit(nopython=True)(pyfunc)
            self.assertPreciseEqual(cfunc(p), pyfunc(p))

        # Homogeneous
        check(Rect(4, 5))
        # Heterogeneous
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

    @tag('important')
    def test_eq(self):
        self._test_compare(eq_usecase)

    @tag('important')
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

    @tag('important')
    def test_getattr(self):
        pyfunc = getattr_usecase
        cfunc = jit(nopython=True)(pyfunc)

        for args in (4, 5, 6), (4, 5.5, 6j):
            p = Point(*args)
            self.assertPreciseEqual(cfunc(p), pyfunc(p))

    @tag('important')
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

    def test_literal_unification(self):
        # Test for #3565.
        @jit(nopython=True)
        def Data1(value):
            return Rect(value, -321)

        @jit(nopython=True)
        def call(i, j):
            if j == 0:
                # In the error, `result` is typed to `Rect(int, LiteralInt)`
                # because of the `-321` literal.  This doesn't match the
                # `result` type in the other branch.
                result = Data1(i)
            else:
                # `result` is typed to be `Rect(int, int)`
                result = Rect(i, j)
            return result

        r = call(123, 1321)
        self.assertEqual(r, Rect(width=123, height=1321))
        r = call(123, 0)
        self.assertEqual(r, Rect(width=123, height=-321))


class TestTupleNRT(TestCase, MemoryLeakMixin):
    def test_tuple_add(self):
        def pyfunc(x):
            a = np.arange(3)
            return (a,) + (x,)

        cfunc = jit(nopython=True)(pyfunc)
        x = 123
        expect_a, expect_x = pyfunc(x)
        got_a, got_x = cfunc(x)
        np.testing.assert_equal(got_a, expect_a)
        self.assertEqual(got_x, expect_x)


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


class TestConversions(TestCase):
    """
    Test implicit conversions between tuple types.
    """

    def check_conversion(self, fromty, toty, val):
        pyfunc = identity
        cr = compile_isolated(pyfunc, (fromty,), toty)
        cfunc = cr.entry_point
        res = cfunc(val)
        self.assertEqual(res, val)

    def test_conversions(self):
        check = self.check_conversion
        fromty = types.UniTuple(types.int32, 2)
        check(fromty, types.UniTuple(types.float32, 2), (4, 5))
        check(fromty, types.Tuple((types.float32, types.int16)), (4, 5))
        aty = types.UniTuple(types.int32, 0)
        bty = types.Tuple(())
        check(aty, bty, ())
        check(bty, aty, ())

        with self.assertRaises(errors.TypingError) as raises:
            check(fromty, types.Tuple((types.float32,)), (4, 5))
        msg = "No conversion from tuple(int32 x 2) to tuple(float32 x 1)"
        self.assertIn(msg, str(raises.exception))


class TestMethods(TestCase):

    def test_index(self):
        pyfunc = index_method_usecase
        cfunc = jit(nopython=True)(pyfunc)
        self.assertEqual(cfunc((1, 2, 3), 2), 1)

        with self.assertRaises(ValueError) as raises:
            cfunc((1, 2, 3), 4)
        msg = 'tuple.index(x): x not in tuple'
        self.assertEqual(msg, str(raises.exception))


class TestTupleBuild(TestCase):

    @unittest.skipIf(utils.PYVERSION < (3, 0), "needs Python 3")
    def test_build_unpack(self):
        def check(p):
            # using eval here since Python 2 doesn't even support the syntax
            pyfunc = eval("lambda a: (1, *a)")
            cfunc = jit(nopython=True)(pyfunc)
            self.assertPreciseEqual(cfunc(p), pyfunc(p))

        # Homogeneous
        check((4, 5))
        # Heterogeneous
        check((4, 5.5))


    @unittest.skipIf(utils.PYVERSION < (3, 0), "needs Python 3")
    def test_build_unpack_more(self):
        def check(p):
            # using eval here since Python 2 doesn't even support the syntax
            pyfunc = eval("lambda a: (1, *a, (1, 2), *a)")
            cfunc = jit(nopython=True)(pyfunc)
            self.assertPreciseEqual(cfunc(p), pyfunc(p))

        # Homogeneous
        check((4, 5))
        # Heterogeneous
        check((4, 5.5))


    @unittest.skipIf(utils.PYVERSION < (3, 0), "needs Python 3")
    def test_build_unpack_call(self):
        def check(p):
            # using eval here since Python 2 doesn't even support the syntax
            @jit
            def inner(*args):
                return args
            pyfunc = eval("lambda a: inner(1, *a)", locals())
            cfunc = jit(nopython=True)(pyfunc)
            self.assertPreciseEqual(cfunc(p), pyfunc(p))

        # Homogeneous
        check((4, 5))
        # Heterogeneous
        check((4, 5.5))

    @unittest.skipIf(utils.PYVERSION < (3, 6), "needs Python 3.6+")
    def test_build_unpack_call_more(self):
        def check(p):
            # using eval here since Python 2 doesn't even support the syntax
            @jit
            def inner(*args):
                return args
            pyfunc = eval("lambda a: inner(1, *a, *(1, 2), *a)", locals())
            cfunc = jit(nopython=True)(pyfunc)
            self.assertPreciseEqual(cfunc(p), pyfunc(p))

        # Homogeneous
        check((4, 5))
        # Heterogeneous
        check((4, 5.5))

    def test_tuple_constructor(self):
        def check(pyfunc, arg):
            cfunc = jit(nopython=True)(pyfunc)
            self.assertPreciseEqual(cfunc(arg), pyfunc(arg))

        # empty
        check(lambda _: tuple(), ())
        # Homogeneous
        check(lambda a: tuple(a), (4, 5))
        # Heterogeneous
        check(lambda a: tuple(a), (4, 5.5))



if __name__ == '__main__':
    unittest.main()
