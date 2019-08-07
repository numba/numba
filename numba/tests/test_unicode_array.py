from __future__ import print_function, unicode_literals

import sys
import numpy as np

import numba.unittest_support as unittest
from numba import jit
from numba.tests.support import TestCase

def getitem(x, i):
    x[i]

def return_getitem(x, i):
    return x[i]

def equal_getitem(x, i, j):
    x[i] == x[j]

def add_getitem(x, i, j):
    x[i] + x[j]

class TestUnicodeArray(TestCase):
    
    def _test(self, pyfunc, *args, **kwargs):
        expected = pyfunc(*args, **kwargs)
        cfunc = jit(nopython=True)(pyfunc)
        self.assertPreciseEqual(cfunc(*args, **kwargs), expected)
    
    def test_getitem(self):
        pyfunc = getitem
        self._test(pyfunc, np.array([1, 2]), 1)
        self._test(pyfunc, '12', 1)
        self._test(pyfunc, b'12', 1)
        self._test(pyfunc, np.array(b'12'), ())
        self._test(pyfunc, np.array('12'), ())
        self._test(pyfunc, np.array([b'12', b'3']), 0)
        self._test(pyfunc, np.array([b'12', b'3']), 1)
        self._test(pyfunc, np.array(['12', '3']), 0)
        self._test(pyfunc, np.array(['12', '3']), 1)

    def test_return_getitem(self):
        pyfunc = return_getitem
        self._test(pyfunc, np.array([1, 2]), 1)
        self._test(pyfunc, '12', 1)
        self._test(pyfunc, b'12', 1)
        self._test(pyfunc, np.array(b'12'), ())
        self._test(pyfunc, np.array('1234'), ())
        self._test(pyfunc, np.array([b'12', b'3']), 0)
        self._test(pyfunc, np.array([b'12', b'3']), 1)
        self._test(pyfunc, np.array(['12', '3']), 0)
        self._test(pyfunc, np.array(['12', '3']), 1)

    def test_equal_getitem(self):
        pyfunc = equal_getitem
        self._test(pyfunc, np.array([1, 2]), 0, 1)
        self._test(pyfunc, '12', 0, 1)
        self._test(pyfunc, b'12', 0, 1)
        #self._test(pyfunc, np.array(b'12'), (), ())  # fails as unsupported
        #self._test(pyfunc, np.array('1234'), (), ())  # fails as unsupported
        #self._test(pyfunc, np.array([b'1', b'2']), 0, 1)  # fails as unsupported
        #self._test(pyfunc, np.array(['1', '2']), 0, 1)  # fails as unsupported

    def test_add_getitem(self):
        pyfunc = add_getitem
        self._test(pyfunc, np.array([1, 2]), 0, 1)
        self._test(pyfunc, '12', 0, 1)
        self._test(pyfunc, b'12', 0, 1)
        #self._test(pyfunc, np.array(b'12'), (), ())  # fails as unsupported
        #self._test(pyfunc, np.array(b'12'), (), ())  # fails as unsupported
        
if __name__ == '__main__':
    unittest.main()
