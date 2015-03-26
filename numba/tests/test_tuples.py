from __future__ import print_function, division, absolute_import

import numpy

from numba import unittest_support as unittest
from numba.compiler import compile_isolated
from numba import types
from .support import TestCase


def tuple_return_usecase(a, b):
    return a, b

def tuple_first(tup):
    a, b = tup
    return a

def tuple_second(tup):
    a, b = tup
    return b

def len_usecase(tup):
    return len(tup)


class TestTupleReturn(TestCase):

    def test_array_tuple(self):
        aryty = types.Array(types.float64, 1, 'C')
        cres = compile_isolated(tuple_return_usecase, (aryty, aryty))
        a = b = numpy.arange(5, dtype='float64')
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


if __name__ == '__main__':
    unittest.main()
