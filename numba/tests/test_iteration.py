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

def tuple_tuple_iter_usecase():
    # Recursively homogenous tuple type
    res = 0.0
    for i in ((1.5, 2.0), (99.3, 3.4), (1.8, 2.5)):
        for j in i:
            res += j
        res = res * 2
    return res

def enumerate_nested_tuple_usecase():
    res = 0.0
    for i, j in enumerate(((1.5, 2.0), (99.3, 3.4), (1.8, 2.5))):
        for l in j:
            res += i * l
        res = res * 2
    return res

def nested_enumerate_usecase():
    res = 0.0
    for i, (j, k) in enumerate(enumerate(((1.5, 2.0), (99.3, 3.4), (1.8, 2.5)))):
        for l in k:
            res += i * j * l
        res = res * 2
    return res


class IterationTest(TestCase):

    def run_nullary_func(self, pyfunc, flags):
        cr = compile_isolated(pyfunc, (), flags=flags)
        cfunc = cr.entry_point
        expected = pyfunc()
        self.assertPreciseEqual(cfunc(), expected)

    def test_int_tuple_iter(self, flags=force_pyobj_flags):
        self.run_nullary_func(int_tuple_iter_usecase, flags)

    def test_int_tuple_iter_npm(self):
        self.test_int_tuple_iter(flags=no_pyobj_flags)

    # Type inference on tuples used to be hardcoded for ints, check
    # that it works for other types.

    def test_float_tuple_iter(self, flags=force_pyobj_flags):
        self.run_nullary_func(float_tuple_iter_usecase, flags)

    def test_float_tuple_iter_npm(self):
        self.test_float_tuple_iter(flags=no_pyobj_flags)

    def test_tuple_tuple_iter(self, flags=force_pyobj_flags):
        self.run_nullary_func(tuple_tuple_iter_usecase, flags)

    def test_tuple_tuple_iter_npm(self):
        self.test_tuple_tuple_iter(flags=no_pyobj_flags)

    def test_enumerate_nested_tuple(self, flags=force_pyobj_flags):
        self.run_nullary_func(enumerate_nested_tuple_usecase, flags)

    def test_enumerate_nested_tuple_npm(self):
        self.test_enumerate_nested_tuple(flags=no_pyobj_flags)

    def test_nested_enumerate(self, flags=force_pyobj_flags):
        self.run_nullary_func(nested_enumerate_usecase, flags)

    def test_nested_enumerate_npm(self):
        self.test_nested_enumerate(flags=no_pyobj_flags)


if __name__ == '__main__':
    unittest.main()
