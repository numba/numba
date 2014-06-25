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


class TestUnpack(TestCase):

    def test_unpack_list(self):
        pyfunc = unpack_list
        cr = compile_isolated(pyfunc, (), flags=enable_pyobj_flags)
        cfunc = cr.entry_point
        l = [1, 2, 3]
        self.assertEqual(cfunc(l), pyfunc(l))

    def test_unpack_shape(self, flags=enable_pyobj_flags):
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

    def test_unpack_range(self, flags=enable_pyobj_flags):
        pyfunc = unpack_range
        cr = compile_isolated(pyfunc, (), flags=flags)
        cfunc = cr.entry_point
        self.assertPreciseEqual(cfunc(), pyfunc())

    def test_unpack_range_npm(self):
        self.test_unpack_range(flags=no_pyobj_flags)


if __name__ == '__main__':
    unittest.main(buffer=True)

