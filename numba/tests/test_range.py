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


if __name__ == '__main__':
    unittest.main()

