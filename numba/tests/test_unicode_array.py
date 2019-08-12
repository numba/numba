from __future__ import print_function, unicode_literals

import numpy as np

import numba.unittest_support as unittest
from numba import jit, utils
from numba.tests.support import TestCase

skip_py2 = unittest.skipIf(not utils.IS_PY3, "not supported in Python 2")


def getitem(x, i):
    return x[i]


def getitem2(x, i, j):
    return x[i][j]


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

    def test_return_isascii(self):
        pyfunc = return_isascii
        cfunc = jit(nopython=True)(pyfunc)
        self._test(pyfunc, cfunc, np.array('1234'), ())
        self._test(pyfunc, cfunc, np.array(['1234']), 0)
        self._test(pyfunc, cfunc, np.array('1234\u00e9'), ())
        self._test(pyfunc, cfunc, np.array(['1234\u00e9']), 0)

    def test_return_str(self):
        pyfunc = return_str
        cfunc = jit(nopython=True)(pyfunc)
        self._test(pyfunc, cfunc, np.array('1234'), ())
        self._test(pyfunc, cfunc, np.array(['1234']), 0)

    def test_hash(self):
        pyfunc = return_str
        cfunc = jit(nopython=True)(pyfunc)
        self._test(pyfunc, cfunc, np.array('1234\u00e9'), ())
        self._test(pyfunc, cfunc, np.array(['1234']), 0)


if __name__ == '__main__':
    unittest.main()
