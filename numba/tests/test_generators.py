from __future__ import print_function
import numba.unittest_support as unittest
from numba.compiler import compile_isolated, Flags
from numba import types, utils
from numba.tests import usecases
from numba.tests.support import TestCase
import numba.unittest_support as unittest
import math
import numpy as np


def generator_func():
    for i in range(10):
        yield i


def return_generator_func(x):
    return (i*2 for i in x)


class TestLists(TestCase):

    @unittest.expectedFailure
    def test_generator_func(self):
        pyfunc = generator_func
        cr = compile_isolated(pyfunc, ())
        cfunc = cr.entry_point
        self.assertEqual([x for x in cfunc()], [x for x in pyfunc()])

    @unittest.expectedFailure
    def test_return_generator_func(self):
        pyfunc = return_generator_func
        cr = compile_isolated(pyfunc, ())
        cfunc = cr.entry_point
        self.assertEqual(sum(cfunc([1, 2, 3])), sum(pyfunc([1, 2, 3])))


if __name__ == '__main__':
    unittest.main()
