from __future__ import print_function
import numba.unittest_support as unittest
import numpy
from numba.compiler import compile_isolated
from numba import types


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


def range1_writeout(n, out):
    i = 0
    for j in range(n):
        out[i] = j
        i += 1
    return i


def range2_writeout(a, b, out):
    i = 0
    for j in range(a, b):
        out[i] = j
        i += 1
    return i


def range3_writeout(a, b, c, out):
    i = 0
    for j in range(a, b, c):
        out[i] = j
        i += 1
    return i


class TestRange(unittest.TestCase):
    def test_loop1_int16(self):
        pyfunc = loop1
        cres = compile_isolated(pyfunc, [types.int16])
        cfunc = cres.entry_point
        self.assertTrue(cfunc(5), pyfunc(5))

    def test_loop2_int16(self):
        pyfunc = loop2
        cres = compile_isolated(pyfunc, [types.int16, types.int16])
        cfunc = cres.entry_point
        self.assertTrue(cfunc(1, 6), pyfunc(1, 6))

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

    def test_range1_writeout_uint64(self):
        pyfunc = range1_writeout
        cres = compile_isolated(pyfunc, [types.uint64, types.int64[:]])
        cfunc = cres.entry_point
        n = 50
        expected = numpy.zeros(n, dtype=numpy.int64)
        got = numpy.zeros(n, dtype=numpy.int64)
        self.assertTrue(cfunc(n, got), pyfunc(n, expected))
        numpy.testing.assert_equal(expected, got)

    def test_range2_writeout_uint64(self):
        pyfunc = range2_writeout
        cres = compile_isolated(pyfunc, [types.uint64, types.uint64,
                                         types.int64[:]])
        cfunc = cres.entry_point
        a, b = 2552764644, 2552764787
        expected = numpy.zeros(b - a, dtype=numpy.int64)
        got = numpy.zeros(b - a, dtype=numpy.int64)
        self.assertTrue(cfunc(a, b, got), pyfunc(a, b, expected))
        numpy.testing.assert_equal(expected, got)

    def test_range3_writeout_uint64(self):
        pyfunc = range3_writeout
        cres = compile_isolated(pyfunc, [types.uint64, types.uint64,
                                         types.uint64, types.int64[:]])
        cfunc = cres.entry_point
        a, b = 2552764644, 2552764787
        c = 1
        expected = numpy.zeros(b - a, dtype=numpy.int64)
        got = numpy.zeros(b - a, dtype=numpy.int64)
        self.assertTrue(cfunc(a, b, c, got), pyfunc(a, b, c, expected))
        numpy.testing.assert_equal(expected, got)

    def test_range1_writeout_uint32(self):
        pyfunc = range1_writeout
        cres = compile_isolated(pyfunc, [types.uint32, types.int64[:]])
        cfunc = cres.entry_point
        n = 50
        expected = numpy.zeros(n, dtype=numpy.int64)
        got = numpy.zeros(n, dtype=numpy.int64)
        self.assertTrue(cfunc(n, got), pyfunc(n, expected))
        numpy.testing.assert_equal(expected, got)

    def test_range2_writeout_uint32(self):
        pyfunc = range2_writeout
        cres = compile_isolated(pyfunc, [types.uint32, types.uint32,
                                         types.int64[:]])
        cfunc = cres.entry_point
        # Test with integer that used the MSB of uint32
        a, b = 0xF0000001, 0xF0000010
        self.assertEqual(a >> 31, 1)
        self.assertEqual(b >> 31, 1)
        expected = numpy.zeros(b - a, dtype=numpy.int64)
        got = numpy.zeros(b - a, dtype=numpy.int64)
        self.assertTrue(cfunc(a, b, got), pyfunc(a, b, expected))
        numpy.testing.assert_equal(expected, got)

    def test_range3_writeout_uint32(self):
        pyfunc = range3_writeout
        cres = compile_isolated(pyfunc, [types.uint32, types.uint32,
                                         types.uint32, types.int64[:]])
        cfunc = cres.entry_point
        # Test with integer that used the MSB of uint32
        a, b = 0xF0000001, 0xF0000010
        self.assertEqual(a >> 31, 1)
        self.assertEqual(b >> 31, 1)
        c = 1
        expected = numpy.zeros(b - a, dtype=numpy.int64)
        got = numpy.zeros(b - a, dtype=numpy.int64)
        self.assertTrue(cfunc(a, b, c, got), pyfunc(a, b, c, expected))
        numpy.testing.assert_equal(expected, got)


if __name__ == '__main__':
    unittest.main()

