from __future__ import print_function, unicode_literals

import numpy as np

import numba.unittest_support as unittest
from numba import jit, utils, from_dtype, types
from numba.typed import Dict
from numba.tests.support import TestCase

skip_py2 = unittest.skipIf(not utils.IS_PY3, "not supported in Python 2")
require_py37 = unittest.skipIf(utils.PYVERSION < (3, 7), "requires Python 3.7+")


def getitem(x, i):
    return x[i]


def getitem2(x, i, j):
    return x[i][j]


def setitem(x, i, v):
    x[i] = v
    return x


def setitem2(x, i, y, j):
    x[i] = y[j]
    return x


def getitem_key(x, y, j):
    x[y[j]] = 123


def return_len(x, i):
    return len(x[i])


def equal_getitem(x, i, j):
    return x[i] == x[j]


def notequal_getitem(x, i, j):
    return x[i] != x[j]


def equal_getitem_value(x, i, v):
    r1 = x[i] == v
    r2 = v == x[i]
    if r1 == r2:
        return r1
    raise ValueError('x[i] == v and v == x[i] are unequal')


def notequal_getitem_value(x, i, v):
    r1 = x[i] != v
    r2 = v != x[i]
    if r1 == r2:
        return r1
    raise ValueError('x[i] != v and v != x[i] are unequal')


def return_isascii(x, i):
    return x[i].isascii()


def return_isupper(x, i):
    return x[i].isupper()


def return_str(x, i):
    return str(x[i])


def return_hash(x, i):
    return hash(x[i])


@skip_py2
class TestUnicodeArray(TestCase):

    def _test(self, pyfunc, cfunc, *args, **kwargs):
        expected = pyfunc(*args, **kwargs)
        self.assertPreciseEqual(cfunc(*args, **kwargs), expected)

    def test_getitem2(self):
        cgetitem2 = jit(nopython=True)(getitem2)

        arr = np.array(b'12')
        self.assertPreciseEqual(cgetitem2(arr, (), 0), getitem2(arr, (), 0))
        with self.assertRaisesRegex(IndexError, 'index out of range'):
            cgetitem2(arr, (), 2)

        arr = np.array('12')
        self.assertPreciseEqual(cgetitem2(arr, (), 0), getitem2(arr, (), 0))
        with self.assertRaisesRegex(IndexError, 'index out of range'):
            cgetitem2(arr, (), 2)

        arr = np.array([b'12', b'3'])
        self.assertPreciseEqual(cgetitem2(arr, 0, 0), getitem2(arr, 0, 0))
        self.assertPreciseEqual(cgetitem2(arr, 0, 1), getitem2(arr, 0, 1))
        self.assertPreciseEqual(cgetitem2(arr, 1, 0), getitem2(arr, 1, 0))
        with self.assertRaisesRegex(IndexError, 'index out of range'):
            cgetitem2(arr, 1, 1)

        arr = np.array(['12', '3'])
        self.assertPreciseEqual(cgetitem2(arr, 0, 0), getitem2(arr, 0, 0))
        self.assertPreciseEqual(cgetitem2(arr, 0, 1), getitem2(arr, 0, 1))
        self.assertPreciseEqual(cgetitem2(arr, 1, 0), getitem2(arr, 1, 0))
        with self.assertRaisesRegex(IndexError, 'index out of range'):
            cgetitem2(arr, 1, 1)

    def test_getitem(self):
        pyfunc = getitem
        cfunc = jit(nopython=True)(pyfunc)
        self._test(pyfunc, cfunc, b'12', 1)
        self._test(pyfunc, cfunc, np.array(b'12'), ())
        self._test(pyfunc, cfunc, np.array([b'12', b'3']), 0)
        self._test(pyfunc, cfunc, np.array([b'12', b'3']), 1)

        self._test(pyfunc, cfunc, '12', 1)
        self._test(pyfunc, cfunc, np.array('12'), ())
        self._test(pyfunc, cfunc, np.array(['12', '3']), 0)
        self._test(pyfunc, cfunc, np.array(['12', '3']), 1)

    def test_getitem_key(self):
        pyfunc = getitem_key
        cfunc = jit(nopython=True)(pyfunc)

        for x, i in [
                (np.array('123'), ()),
                (np.array(['123']), 0),
                (np.array(b'123'), ()),
                (np.array([b'123']), 0)
        ]:
            d1 = {}
            d2 = Dict.empty(from_dtype(x.dtype), types.int64)
            pyfunc(d1, x, i)
            cfunc(d2, x, i)
            self.assertEqual(d1, d2)
            # check for charseq to str conversion:
            str(d2)

    def test_setitem(self):
        pyfunc = setitem
        cfunc = jit(nopython=True)(pyfunc)

        x = np.array(12)
        self._test(pyfunc, cfunc, x, (), 34)

        x1 = np.array(b'123')
        x2 = np.array(b'123')
        y1 = pyfunc(x1, (), b'34')
        y2 = cfunc(x2, (), b'34')
        self.assertPreciseEqual(x1, x2)
        self.assertPreciseEqual(y1, y2)

        x1 = np.array(['123'])
        x2 = np.array(['123'])
        y1 = pyfunc(x1, 0, '34')
        y2 = cfunc(x2, 0, '34')
        self.assertPreciseEqual(x1, x2)
        self.assertPreciseEqual(y1, y2)

    def test_setitem2(self):
        pyfunc = setitem2
        cfunc = jit(nopython=True)(pyfunc)

        x1 = np.array(['123', 'ABC'])
        x2 = np.array(['123', 'ABC'])
        y1 = pyfunc(x1, 0, x1, 1)
        y2 = cfunc(x2, 0, x2, 1)
        self.assertPreciseEqual(x1, x2)
        self.assertPreciseEqual(y1, y2)

        x1 = np.array([b'123', b'ABC'])
        x2 = np.array([b'123', b'ABC'])
        y1 = pyfunc(x1, 0, x1, 1)
        y2 = cfunc(x2, 0, x2, 1)
        self.assertPreciseEqual(x1, x2)
        self.assertPreciseEqual(y1, y2)

        x1 = np.array('123')
        x2 = np.array('123')
        z1 = np.array('ABC')
        z2 = np.array('ABC')
        y1 = pyfunc(x1, (), z1, ())
        y2 = cfunc(x2, (), z2, ())
        self.assertPreciseEqual(x1, x2)
        self.assertPreciseEqual(y1, y2)

        x1 = np.array(123)
        x2 = np.array(123)
        z1 = 456,
        z2 = 456,
        y1 = pyfunc(x1, (), z1, 0)
        y2 = cfunc(x2, (), z2, 0)
        self.assertPreciseEqual(x1, x2)
        self.assertPreciseEqual(y1, y2)

        # bytes
        x1 = np.array(b'123')
        x2 = np.array(b'123')
        z1 = b'ABC',
        z2 = b'ABC',
        y1 = pyfunc(x1, (), z1, 0)
        y2 = cfunc(x2, (), z2, 0)
        self.assertPreciseEqual(x1, x2)
        self.assertPreciseEqual(y1, y2)

        # UTF-8
        x1 = np.array('123')
        x2 = np.array('123')
        z1 = 'ABC',
        z2 = 'ABC',
        y1 = pyfunc(x1, (), z1, 0)
        y2 = cfunc(x2, (), z2, 0)
        self.assertPreciseEqual(x1, x2)
        self.assertPreciseEqual(y1, y2)

        # UTF-16
        x1 = np.array('123')
        x2 = np.array('123')
        z1 = 'AB\u01e9',
        z2 = 'AB\u01e9',
        y1 = pyfunc(x1, (), z1, 0)
        y2 = cfunc(x2, (), z2, 0)
        self.assertPreciseEqual(x1, x2)
        self.assertPreciseEqual(y1, y2)

        # UTF-32
        x1 = np.array('123')
        x2 = np.array('123')
        z1 = 'AB\U00108a0e',
        z2 = 'AB\U00108a0e',
        y1 = pyfunc(x1, (), z1, 0)
        y2 = cfunc(x2, (), z2, 0)
        self.assertPreciseEqual(x1, x2)
        self.assertPreciseEqual(y1, y2)

        # UTF-8, assign longer value (truncates as in numpy)
        x1 = np.array('123')
        x2 = np.array('123')
        z1 = 'ABCD',
        z2 = 'ABCD',
        y1 = pyfunc(x1, (), z1, 0)
        y2 = cfunc(x2, (), z2, 0)
        self.assertPreciseEqual(x1, x2)
        self.assertPreciseEqual(y1, y2)

        # UTF-8, assign shorter value
        x1 = np.array('123')
        x2 = np.array('123')
        z1 = 'AB',
        z2 = 'AB',
        y1 = pyfunc(x1, (), z1, 0)
        y2 = cfunc(x2, (), z2, 0)
        self.assertPreciseEqual(x1, x2)
        self.assertPreciseEqual(y1, y2)

        # bytes, assign longer value (truncates as in numpy)
        x1 = np.array(b'123')
        x2 = np.array(b'123')
        z1 = b'ABCD',
        z2 = b'ABCD',
        y1 = pyfunc(x1, (), z1, 0)
        y2 = cfunc(x2, (), z2, 0)
        self.assertPreciseEqual(x1, x2)
        self.assertPreciseEqual(y1, y2)

        # bytes, assign shorter value
        x1 = np.array(b'123')
        x2 = np.array(b'123')
        z1 = b'AB',
        z2 = b'AB',
        y1 = pyfunc(x1, (), z1, 0)
        y2 = cfunc(x2, (), z2, 0)
        self.assertPreciseEqual(x1, x2)
        self.assertPreciseEqual(y1, y2)

    def test_return_len(self):
        pyfunc = return_len
        cfunc = jit(nopython=True)(pyfunc)
        self._test(pyfunc, cfunc, np.array(b'12'), ())
        self._test(pyfunc, cfunc, np.array('12'), ())
        self._test(pyfunc, cfunc, np.array([b'12', b'3']), 0)
        self._test(pyfunc, cfunc, np.array(['12', '3']), 0)
        self._test(pyfunc, cfunc, np.array([b'12', b'3']), 1)
        self._test(pyfunc, cfunc, np.array(['12', '3']), 1)

    def _test_op_getitem(self, pyfunc):
        cfunc = jit(nopython=True)(pyfunc)
        self._test(pyfunc, cfunc, np.array([1, 2]), 0, 1)
        self._test(pyfunc, cfunc, '12', 0, 1)
        self._test(pyfunc, cfunc, b'12', 0, 1)
        self._test(pyfunc, cfunc, np.array(b'12'), (), ())
        self._test(pyfunc, cfunc, np.array('1234'), (), ())

        self._test(pyfunc, cfunc, np.array([b'1', b'2']), 0, 0)
        self._test(pyfunc, cfunc, np.array([b'1', b'2']), 0, 1)
        self._test(pyfunc, cfunc, np.array([b'12', b'3']), 0, 0)
        self._test(pyfunc, cfunc, np.array([b'12', b'3']), 1, 1)
        self._test(pyfunc, cfunc, np.array([b'12', b'3']), 0, 1)
        self._test(pyfunc, cfunc, np.array([b'12', b'3']), 1, 0)

        self._test(pyfunc, cfunc, np.array(['1', '2']), 0, 0)
        self._test(pyfunc, cfunc, np.array(['1', '2']), 0, 1)
        self._test(pyfunc, cfunc, np.array(['12', '3']), 0, 0)
        self._test(pyfunc, cfunc, np.array(['12', '3']), 1, 1)
        self._test(pyfunc, cfunc, np.array(['12', '3']), 0, 1)
        self._test(pyfunc, cfunc, np.array(['12', '3']), 1, 0)

    def test_equal_getitem(self):
        self._test_op_getitem(equal_getitem)

    def test_notequal_getitem(self):
        self._test_op_getitem(notequal_getitem)

    def _test_op_getitem_value(self, pyfunc):
        cfunc = jit(nopython=True)(pyfunc)
        self._test(pyfunc, cfunc, np.array([1, 2]), 0, 1)
        self._test(pyfunc, cfunc, '12', 0, '1')
        self._test(pyfunc, cfunc, '12', 1, '3')
        self._test(pyfunc, cfunc, np.array('1234'), (), '1234')
        self._test(pyfunc, cfunc, np.array(['1234']), 0, '1234')
        self._test(pyfunc, cfunc, np.array(['1234']), 0, 'abc')
        #self._test(pyfunc, cfunc, b'12', 0, b'1')  # fails: No conversion from array(bool, 1d, C) to bool
        self._test(pyfunc, cfunc, np.array(b'12'), (), b'12')
        self._test(pyfunc, cfunc, np.array([b'12']), 0, b'12')
        self._test(pyfunc, cfunc, np.array([b'12']), 0, b'a')

    def test_equal_getitem_value(self):
        self._test_op_getitem_value(equal_getitem_value)

    def test_notequal_getitem_value(self):
        self._test_op_getitem_value(notequal_getitem_value)

    @require_py37
    def test_return_isascii(self):
        pyfunc = return_isascii
        cfunc = jit(nopython=True)(pyfunc)
        self._test(pyfunc, cfunc, np.array('1234'), ())
        self._test(pyfunc, cfunc, np.array(['1234']), 0)
        self._test(pyfunc, cfunc, np.array('1234\u00e9'), ())
        self._test(pyfunc, cfunc, np.array(['1234\u00e9']), 0)

    def test_return_isupper(self):
        pyfunc = return_isupper
        cfunc = jit(nopython=True)(pyfunc)
        self._test(pyfunc, cfunc, np.array('abc'), ())
        self._test(pyfunc, cfunc, np.array(['abc']), 0)

    def test_return_str(self):
        pyfunc = return_str
        cfunc = jit(nopython=True)(pyfunc)
        self._test(pyfunc, cfunc, np.array('1234'), ())
        self._test(pyfunc, cfunc, np.array(['1234']), 0)

    def test_hash(self):
        pyfunc = return_hash
        cfunc = jit(nopython=True)(pyfunc)

        assert pyfunc(np.array('123'), ()) == hash('123') == hash(np.array('123')[()])

        self._test(pyfunc, cfunc, np.array('1234'), ())
        self._test(pyfunc, cfunc, np.array(['1234']), 0)

        self._test(pyfunc, cfunc, np.array('1234\u00e9'), ())
        self._test(pyfunc, cfunc, np.array(['1234u00e9']), 0)

        self._test(pyfunc, cfunc, np.array('1234\U00108a0e'), ())
        self._test(pyfunc, cfunc, np.array(['1234\U00108a0e']), 0)

        self._test(pyfunc, cfunc, np.array(b'1234'), ())
        self._test(pyfunc, cfunc, np.array([b'1234']), 0)


if __name__ == '__main__':
    unittest.main()
