from __future__ import print_function, division, absolute_import
import numpy
from numba import unittest_support as unittest
from numba.compiler import compile_isolated
from numba import types


def tuple_return_usecase(a, b):
    return a, b


class TestTupleReturn(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
