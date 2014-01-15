
from __future__ import print_function
import numba.unittest_support as unittest
import numpy
from numba.compiler import compile_isolated, Flags
from numba import types

enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

force_pyobj_flags = Flags()
force_pyobj_flags.set("force_pyobject")


def unpack_list(l):
    a, b, c = l
    return (a, b, c)


def unpack_shape(a):
    x, y, z = a.shape
    return x + y + z


class TestUnpack(unittest.TestCase):
    def test_unpack_list(self):
        pyfunc = unpack_list
        cr = compile_isolated(pyfunc, (), flags=enable_pyobj_flags)
        cfunc = cr.entry_point
        l = [1, 2, 3]
        self.assertEqual(cfunc(l), pyfunc(l))

    def test_unpack_shape(self):
        pyfunc = unpack_shape
        cr = compile_isolated(pyfunc, [types.Array(dtype=types.int32,
                                                        ndim=3,
                                                        layout='C')])
        cfunc = cr.entry_point
        a = numpy.zeros(shape=(1, 2, 3))
        self.assertEqual(cfunc(a), pyfunc(a))


if __name__ == '__main__':
    unittest.main(buffer=True)

