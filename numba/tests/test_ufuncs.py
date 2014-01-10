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


# unary ufuncs
def absolute_ufunc(x, result):
    np.absolute(x, result)

def exp_ufunc(x, result):
    np.exp(x, result)

def sin_ufunc(x, result):
    np.sin(x, result)

def cos_ufunc(x, result):
    np.cos(x, result)

def tan_ufunc(x, result):
    np.tan(x, result)

# binary ufuncs
def add_ufunc(x, y, result):
    np.add(x, y, result)

def subtract_ufunc(x, y, result):
    np.subtract(x, y, result)

def multiply_ufunc(x, y, result):
    np.multiply(x, y, result)

def divide_ufunc(x, y, result):
    np.divide(x, y, result)


class TestUFuncs(unittest.TestCase):

    def test_unary_ufuncs(self):
        ufunc_list = [absolute_ufunc, exp_ufunc, sin_ufunc, cos_ufunc,
                      tan_ufunc]

        arraytypes = [types.Array(types.int32, 1, 'C'),
                      types.Array(types.int64, 1, 'C'),
                      types.Array(types.float32, 1, 'C'),
                      types.Array(types.float64, 1, 'C')]

        x_operands = [np.arange(-10, 10, dtype='i4'),
                      np.arange(-10, 10, dtype='i8'),
                      np.arange(-1, 1, 0.1, dtype='f4'),
                      np.arange(-1, 1, 0.1, dtype='f8')]

        for arraytype, x_operand in zip(arraytypes, x_operands):
            for ufunc in ufunc_list:
                pyfunc = ufunc
                cr = compile_isolated(pyfunc, (arraytype, arraytype))
                cfunc = cr.entry_point

                result = np.zeros(x_operand.size, dtype=x_operand.dtype)
                cfunc(x_operand, result)
                control = np.zeros(x_operand.size, dtype=x_operand.dtype)
                ufunc(x_operand, control)
                self.assertTrue(np.allclose(result, control))

    def test_binary_ufuncs(self):
        ufunc_list = [add_ufunc, subtract_ufunc, multiply_ufunc, divide_ufunc]

        arraytypes = [types.Array(types.int32, 1, 'C'),
                      types.Array(types.int64, 1, 'C'),
                      types.Array(types.float32, 1, 'C'),
                      types.Array(types.float64, 1, 'C')]

        xy_operands = [np.arange(-10, 10, dtype='i4'),
                       np.arange(-10, 10, dtype='i8'),
                       np.arange(-1, 1, 0.1, dtype='f4'),
                       np.arange(-1, 1, 0.1, dtype='f8')]

        for arraytype, xy_operand in zip(arraytypes, xy_operands):
            for ufunc in ufunc_list:
                pyfunc = ufunc
                cr = compile_isolated(pyfunc, (arraytype, arraytype, arraytype))
                cfunc = cr.entry_point

                result = np.zeros(xy_operand.size, dtype=xy_operand.dtype)
                cfunc(xy_operand, xy_operand, result)
                control = np.zeros(xy_operand.size, dtype=xy_operand.dtype)
                ufunc(xy_operand, xy_operand, control) 
                self.assertTrue((result == control).all())


if __name__ == '__main__':
    unittest.main()

