from __future__ import print_function

import numpy

import numba.unittest_support as unittest
from numba.compiler import compile_isolated, Flags
from numba import types
from .support import TestCase

enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

force_pyobj_flags = Flags()
force_pyobj_flags.set("force_pyobject")

no_pyobj_flags = Flags()


def int_tuple_iter_usecase():
    res = 0
    for i in (1, 2, 99, 3):
        res += i
    return res

def float_tuple_iter_usecase():
    res = 0.0
    for i in (1.5, 2.0, 99.3, 3.4):
        res += i
    return res



class IterationTest(TestCase):

    def test_int_tuple_iter(self, flags=force_pyobj_flags):
        pyfunc = int_tuple_iter_usecase
        cr = compile_isolated(pyfunc, (), flags=flags)
        cfunc = cr.entry_point
        self.assertPreciseEqual(cfunc(), pyfunc())

    def test_int_tuple_iter_npm(self):
        self.test_int_tuple_iter(flags=no_pyobj_flags)

    def test_float_tuple_iter(self, flags=force_pyobj_flags):
        # Type inference on tuples used to be hardcoded for ints, check
        # that it works for other types.
        pyfunc = float_tuple_iter_usecase
        cr = compile_isolated(pyfunc, (), flags=flags)
        cfunc = cr.entry_point
        self.assertPreciseEqual(cfunc(), pyfunc())

    def test_float_tuple_iter_npm(self):
        self.test_float_tuple_iter(flags=no_pyobj_flags)


if __name__ == '__main__':
    unittest.main()
