from __future__ import print_function, division, absolute_import
import numpy
from numba.compiler import compile_isolated
from numba import typeof
from numba import unittest_support as unittest


def array_return(a, i):
    a[i] = 123
    return a


class TestArrayReturn(unittest.TestCase):
    def test_array_return(self):
        a = numpy.arange(10)
        i = 2
        at, it = typeof(a), typeof(i)
        cres = compile_isolated(array_return, (at, it))
        cfunc = cres.entry_point
        self.assertIs(a, cfunc(a, i))

if __name__ == '__main__':
    unittest.main()
