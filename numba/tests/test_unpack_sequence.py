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


def unpack_list(l):
    a, b, c = l
    return (a, b, c)


def unpack_shape(a):
    x, y, z = a.shape
    return x + y + z


def unpack_range():
    a, b, c = range(3)
    return a + b + c


def unpack_range_too_small():
    a, b, c = range(2)
    return a + b + c


def unpack_range_too_large():
    a, b, c = range(4)
    return a + b + c


def unpack_tuple():
    a, b, c = (1, 2, 3)
    return a + b + c


def unpack_tuple_too_small():
    a, b, c = (1, 2)
    return a + b + c


def unpack_tuple_too_large():
    a, b, c = (1, 2, 3, 4)
    return a + b + c


def unpack_heterogenous_tuple():
    a, b, c = (1, 2.5, 3j)
    return a + b + c


def unpack_nested_heterogenous_tuple():
    a, (b, c) = (1, (2.5, 3j))
    return a + b + c


def chained_unpack_assign1(x, y):
    # Used to fail in object mode (issue #580)
    a = (b, c) = (x, y)
    (d, e) = a
    return d + e + b + c


class TestUnpack(TestCase):

    def test_unpack_list(self):
        pyfunc = unpack_list
        cr = compile_isolated(pyfunc, (), flags=force_pyobj_flags)
        cfunc = cr.entry_point
        l = [1, 2, 3]
        self.assertEqual(cfunc(l), pyfunc(l))

    def test_unpack_shape(self, flags=force_pyobj_flags):
        pyfunc = unpack_shape
        cr = compile_isolated(pyfunc, [types.Array(dtype=types.int32,
                                                        ndim=3,
                                                        layout='C')],
                              flags=flags)
        cfunc = cr.entry_point
        a = numpy.zeros(shape=(1, 2, 3))
        self.assertPreciseEqual(cfunc(a), pyfunc(a))

    def test_unpack_shape_npm(self):
        self.test_unpack_shape(flags=no_pyobj_flags)

    def test_unpack_range(self, flags=force_pyobj_flags):
        self.run_nullary_func(unpack_range, flags)

    def test_unpack_range_npm(self):
        self.test_unpack_range(flags=no_pyobj_flags)

    def test_unpack_tuple(self, flags=force_pyobj_flags):
        self.run_nullary_func(unpack_tuple, flags)

    def test_unpack_tuple_npm(self):
        self.test_unpack_tuple(flags=no_pyobj_flags)

    def test_unpack_heterogenous_tuple(self, flags=force_pyobj_flags):
        self.run_nullary_func(unpack_heterogenous_tuple, flags)

    def test_unpack_heterogenous_tuple_npm(self):
        self.test_unpack_heterogenous_tuple(flags=no_pyobj_flags)

    def test_unpack_nested_heterogenous_tuple(self, flags=force_pyobj_flags):
        self.run_nullary_func(unpack_nested_heterogenous_tuple, flags)

    def test_unpack_nested_heterogenous_tuple_npm(self):
        self.test_unpack_nested_heterogenous_tuple(flags=no_pyobj_flags)

    def test_chained_unpack_assign(self, flags=force_pyobj_flags):
        pyfunc = chained_unpack_assign1
        cr = compile_isolated(pyfunc, [types.int32, types.int32],
                              flags=flags)
        cfunc = cr.entry_point
        args = (4, 5)
        self.assertPreciseEqual(cfunc(*args), pyfunc(*args))

    def test_chained_unpack_assign_npm(self):
        self.test_chained_unpack_assign(flags=no_pyobj_flags)

    def check_unpack_error(self, pyfunc, flags=force_pyobj_flags):
        cr = compile_isolated(pyfunc, (), flags=flags)
        cfunc = cr.entry_point
        with self.assertRaises(ValueError):
            cfunc()

    def test_unpack_tuple_too_small(self):
        self.check_unpack_error(unpack_tuple_too_small)

    def test_unpack_tuple_too_small_npm(self):
        self.check_unpack_error(unpack_tuple_too_small, no_pyobj_flags)

    def test_unpack_tuple_too_large(self):
        self.check_unpack_error(unpack_tuple_too_large)

    def test_unpack_tuple_too_large_npm(self):
        self.check_unpack_error(unpack_tuple_too_large, no_pyobj_flags)

    def test_unpack_range_too_small(self):
        self.check_unpack_error(unpack_range_too_small)

    def test_unpack_range_too_small_npm(self):
        self.check_unpack_error(unpack_range_too_small, no_pyobj_flags)

    def test_unpack_range_too_large(self):
        self.check_unpack_error(unpack_range_too_large)

    def test_unpack_range_too_large_npm(self):
        self.check_unpack_error(unpack_range_too_large, no_pyobj_flags)


if __name__ == '__main__':
    unittest.main(buffer=True)

