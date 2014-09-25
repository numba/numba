from __future__ import print_function
import numba.unittest_support as unittest
from numba.compiler import compile_isolated, Flags
from numba import types, utils
from numba.tests import usecases
from numba.tests.support import TestCase
import numba.unittest_support as unittest
import math
import numpy as np


def del_list_item_func(x):
    del x[0]
    return x


def del_ref_func(x):
    del x
    return x


class TestLists(TestCase):

    @unittest.expectedFailure
    def test_del_list_item_func(self):
        pyfunc = del_list_item_func
        cr = compile_isolated(pyfunc, ())
        cfunc = cr.entry_point
        expected = pyfunc([1, 2, 3])
        result = cfunc([1, 2, 3])
        self.assertEqual(expected, result)

    @unittest.expectedFailure
    def test_del_ref_func(self):
        pyfunc = del_ref_func
        cr = compile_isolated(pyfunc, ())
        cfunc = cr.entry_point
        expected = pyfunc(1)
        result = cfunc(1)
        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
