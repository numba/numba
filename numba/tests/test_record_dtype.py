from __future__ import print_function, division, absolute_import

import numpy as np
from numba import numpy_support, types
from numba import unittest_support as unittest
from numba.compiler import compile_isolated
from numba.utils import IS_PY3


def get_a(ary, i):
    return ary[i].a


def get_b(ary, i):
    return ary[i].b


def get_c(ary, i):
    return ary[i].c


def set_a(ary, i, v):
    ary[i].a = v


def set_b(ary, i, v):
    ary[i].b = v


def set_c(ary, i, v):
    ary[i].c = v


recordtype = np.dtype([('a', np.float64),
                       ('b', np.int32),
                       ('c', np.complex64),
                       ('d', (np.str, 5))])


class TestRecordDtype(unittest.TestCase):
    def setUp(self):
        ary = np.recarray(3, dtype=recordtype)
        self.sample1d = ary

        for i in range(ary.size):
            x = i + 1
            ary[i].a = x / 2
            ary[i].b = x
            ary[i].c = x * 1j
            ary[i].d = "%d" % x

    def test_from_dtype(self):
        rec = numpy_support.from_dtype(recordtype)
        self.assertEqual(rec.typeof('a'), types.float64)
        self.assertEqual(rec.typeof('b'), types.int32)
        self.assertEqual(rec.typeof('c'), types.complex64)
        if IS_PY3:
            self.assertEqual(rec.typeof('d'), types.UnicodeCharSeq(5))
        else:
            self.assertEqual(rec.typeof('d'), types.CharSeq(5))
        self.assertEqual(rec.offset('a'), recordtype.fields['a'][1])
        self.assertEqual(rec.offset('b'), recordtype.fields['b'][1])
        self.assertEqual(rec.offset('c'), recordtype.fields['c'][1])
        self.assertEqual(rec.offset('d'), recordtype.fields['d'][1])
        self.assertEqual(recordtype.itemsize, rec.size)

    def _test_get_equal(self, pyfunc):
        rec = numpy_support.from_dtype(recordtype)
        cres = compile_isolated(pyfunc, (rec[:], types.intp))
        cfunc = cres.entry_point
        for i in range(self.sample1d.size):
            self.assertEqual(pyfunc(self.sample1d, i), cfunc(self.sample1d, i))

    def test_get_a(self):
        self._test_get_equal(get_a)

    def test_get_b(self):
        self._test_get_equal(get_b)

    def test_get_c(self):
        self._test_get_equal(get_c)

    def _test_set_equal(self, pyfunc, value, valuetype):
        rec = numpy_support.from_dtype(recordtype)
        cres = compile_isolated(pyfunc, (rec[:], types.intp, valuetype))
        cfunc = cres.entry_point

        for i in range(self.sample1d.size):
            expect = self.sample1d.copy()
            pyfunc(expect, i, value)

            got = self.sample1d.copy()
            cfunc(got, i, value)

            # Match the entire array to ensure no memory corruption
            self.assertTrue(np.all(expect == got))

    def test_set_a(self):
        self._test_set_equal(set_a, 3.1415, types.float64)

    def test_set_b(self):
        self._test_set_equal(set_b, 123, types.int32)

    def test_set_c(self):
        self._test_set_equal(set_c, 43j, types.complex64)


if __name__ == '__main__':
    unittest.main()
