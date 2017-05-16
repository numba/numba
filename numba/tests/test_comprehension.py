from __future__ import print_function

import numba.unittest_support as unittest

import sys

import numpy as np

from numba.compiler import compile_isolated
from numba import types, utils
from .support import tag


def comp_list(n):
    l = [ i for i in range(n) ]
    s = 0
    for i in l:
        s += i
    return s

def comp_with_array(n):
    m = n * 2
    l = np.array([ i + m for i in range(n) ])
    return np.sum(l)

def comp_nest_with_array(n):
    l = np.array([ np.sum(np.array([ i * j for j in range(n) ])) for i in range(n) ])
    return np.sum(l)

class TestListComprehension(unittest.TestCase):

    @tag('important')
    def test_comp_list(self):
        pyfunc = comp_list
        cres = compile_isolated(pyfunc, [types.intp])
        cfunc = cres.entry_point
        self.assertEqual(cfunc(5), pyfunc(5))
        self.assertEqual(cfunc(0), pyfunc(0))
        self.assertEqual(cfunc(-1), pyfunc(-1))

    @tag('important')
    def test_comp_with_array(self):
        pyfunc = comp_with_array
        cres = compile_isolated(pyfunc, [types.intp])
        cfunc = cres.entry_point
        self.assertEqual(cfunc(5), pyfunc(5))

    @tag('important')
    def test_comp_nest_with_array(self):
        pyfunc = comp_nest_with_array
        cres = compile_isolated(pyfunc, [types.intp])
        cfunc = cres.entry_point
        self.assertEqual(cfunc(5), pyfunc(5))

if __name__ == '__main__':
    unittest.main()

