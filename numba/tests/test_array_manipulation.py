from __future__ import print_function
import unittest
import numpy as np
from numba.compiler import compile_isolated, Flags
from numba import types, utils
from numba.tests import usecases

enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

force_pyobj_flags = Flags()
force_pyobj_flags.set("force_pyobject")


def reshape_array(a, control):
    return (a.reshape(3, 3) == control).all()

def flatten_array(a, control):
    return (a.flatten() == control).all()

def transpose_array(a, control):
    return (a.transpose() == control).all()

def convert_array(a, control):
    return (a.astype(dtype='f4') == control).all()


class TestArrayManipulation(unittest.TestCase):

    def test_reshape_array(self):
        pyfunc = reshape_array
        arraytype1 = types.Array(types.int32, 1, 'C')
        arraytype2 = types.Array(types.int32, 2, 'C')
        cr = compile_isolated(pyfunc, (arraytype1, arraytype2),
                              flags=enable_pyobj_flags)
        cfunc = cr.entry_point

        a = np.arange(9)
        control = np.arange(9).reshape(3, 3)
        self.assertTrue(cfunc(a, control))

    def test_flatten_array(self):
        pyfunc = flatten_array
        arraytype1 = types.Array(types.int32, 2, 'C')
        arraytype2 = types.Array(types.int32, 1, 'C')
        cr = compile_isolated(pyfunc, (arraytype1, arraytype2),
                              flags=enable_pyobj_flags)
        cfunc = cr.entry_point

        a = np.arange(9).reshape(3, 3)
        control = np.arange(9).reshape(3, 3).flatten()
        self.assertTrue(cfunc(a, control))

    def test_transpose_array(self):
        pyfunc = transpose_array
        arraytype1 = types.Array(types.int32, 2, 'C')
        arraytype2 = types.Array(types.int32, 2, 'C')
        cr = compile_isolated(pyfunc, (arraytype1, arraytype2),
                              flags=enable_pyobj_flags)
        cfunc = cr.entry_point

        a = np.arange(9).reshape(3, 3)
        control = np.arange(9).reshape(3, 3).transpose()
        self.assertTrue(cfunc(a, control))

    def test_convert_array(self):
        pyfunc = convert_array
        arraytype1 = types.Array(types.int32, 1, 'C')
        arraytype2 = types.Array(types.float32, 1, 'C')
        cr = compile_isolated(pyfunc, (arraytype1, arraytype2),
                              flags=enable_pyobj_flags)
        cfunc = cr.entry_point

        a = np.arange(9, dtype='i4')
        control = np.arange(9, dtype='f4')
        self.assertTrue(cfunc(a, control))

if __name__ == '__main__':
    unittest.main()

