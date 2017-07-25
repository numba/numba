from __future__ import print_function

import numba.unittest_support as unittest

import sys

import numpy

from numba.compiler import compile_isolated
from numba import types
from .support import tag


def loop1(n):
    s = 0
    for i in range(n):
        s += i
    return s


def loop2(a, b):
    s = 0
    for i in range(a, b):
        s += i
    return s


def loop3(a, b, c):
    s = 0
    for i in range(a, b, c):
        s += i
    return s


def xrange_usecase(n):
    s = 0
    for i in xrange(n):
        s += i
    return s

def range_len1(n):
    return len(range(n))

def range_len2(a, b):
    return len(range(a, b))

def range_len3(a, b, c):
    return len(range(a, b, c))

from numba.targets.rangeobj import range_iter_len
def range_iter_len1(a):
    return range_iter_len(iter(range(a)))

def range_iter_len2(a):
    return range_iter_len(iter(a))

class TestRange(unittest.TestCase):

    @tag('important')
    def test_loop1_int16(self):
        pyfunc = loop1
        cres = compile_isolated(pyfunc, [types.int16])
        cfunc = cres.entry_point
        self.assertTrue(cfunc(5), pyfunc(5))

    @tag('important')
    def test_loop2_int16(self):
        pyfunc = loop2
        cres = compile_isolated(pyfunc, [types.int16, types.int16])
        cfunc = cres.entry_point
        self.assertTrue(cfunc(1, 6), pyfunc(1, 6))

    @tag('important')
    def test_loop3_int32(self):
        pyfunc = loop3
        cres = compile_isolated(pyfunc, [types.int32] * 3)
        cfunc = cres.entry_point
        arglist = [
            (1, 2, 1),
            (2, 8, 3),
            (-10, -11, -10),
            (-10, -10, -2),
        ]
        for args in arglist:
            self.assertEqual(cfunc(*args), pyfunc(*args))

    @tag('important')
    @unittest.skipIf(sys.version_info >= (3,), "test is Python 2-specific")
    def test_xrange(self):
        pyfunc = xrange_usecase
        cres = compile_isolated(pyfunc, (types.int32,))
        cfunc = cres.entry_point
        self.assertEqual(cfunc(5), pyfunc(5))

    @tag('important')
    def test_range_len1(self):
        pyfunc = range_len1
        typelist = [types.int16, types.int32, types.int64]
        arglist = [5, 0, -5]
        for typ in typelist:
            cres = compile_isolated(pyfunc, [typ])
            cfunc = cres.entry_point
            for arg in arglist:
                self.assertEqual(cfunc(typ(arg)), pyfunc(typ(arg)))

    @tag('important')
    def test_range_len2(self):
        pyfunc = range_len2
        typelist = [types.int16, types.int32, types.int64]
        arglist = [(1,6), (6,1), (-5, -1)]
        for typ in typelist:
            cres = compile_isolated(pyfunc, [typ] * 2)
            cfunc = cres.entry_point
            for args in arglist:
                args_ = tuple(typ(x) for x in args)
                self.assertEqual(cfunc(*args_), pyfunc(*args_))

    @tag('important')
    def test_range_len3(self):
        pyfunc = range_len3
        typelist = [types.int16, types.int32, types.int64]
        arglist = [
            (1, 2, 1),
            (2, 8, 3),
            (-10, -11, -10),
            (-10, -10, -2),
        ]
        for typ in typelist:
            cres = compile_isolated(pyfunc, [typ] * 3)
            cfunc = cres.entry_point
            for args in arglist:
                args_ = tuple(typ(x) for x in args)
                self.assertEqual(cfunc(*args_), pyfunc(*args_))
 
    @tag('important')
    def test_range_iter_len1(self):
        range_func = range_len1
        range_iter_func = range_iter_len1
        typelist = [types.int16, types.int32, types.int64]
        arglist = [5, 0, -5]
        for typ in typelist:
            cres = compile_isolated(range_iter_func, [typ])
            cfunc = cres.entry_point
            for arg in arglist:
                self.assertEqual(cfunc(typ(arg)), range_func(typ(arg)))

    @tag('important')
    def test_range_iter_list(self):
        range_iter_func = range_iter_len2
        cres = compile_isolated(range_iter_func, [types.List(types.intp)])
        cfunc = cres.entry_point
        arglist = [1, 2, 3, 4, 5]
        self.assertEqual(cfunc(arglist), len(arglist))

if __name__ == '__main__':
    unittest.main()

