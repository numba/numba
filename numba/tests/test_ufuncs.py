from __future__ import print_function
import numba.unittest_support as unittest
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
        ufunc_list = [add_ufunc, subtract_ufunc, multiply_ufunc]
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
                self.assertTrue(np.allclose(result, control))

    def test_binary_ufunc_performance(self):

        pyfunc = add_ufunc
        arraytype = types.Array(types.float32, 1, 'C')
        cr = compile_isolated(pyfunc, (arraytype, arraytype, arraytype))
        cfunc = cr.entry_point

        nelem = 5000
        x_operand = np.arange(nelem, dtype=np.float32)
        y_operand = np.arange(nelem, dtype=np.float32)
        control = np.empty_like(x_operand)
        result = np.empty_like(x_operand)

        def bm_python():
            pyfunc(x_operand, y_operand, control)

        def bm_numba():
            cfunc(x_operand, y_operand, result)

        print(utils.benchmark(bm_python, maxct=1000))
        print(utils.benchmark(bm_numba, maxct=1000))
        assert np.allclose(control, result)

    def test_divide_ufuncs(self):
        ufunc_list = [divide_ufunc]

        arraytypes = [types.Array(types.int32, 1, 'C'),
                      types.Array(types.int64, 1, 'C'),
                      types.Array(types.float32, 1, 'C'),
                      types.Array(types.float64, 1, 'C')]

        x_operands = [np.arange(-10, 10, dtype='i4'),
                      np.arange(-10, 10, dtype='i8'),
                      np.arange(-1, 1, 0.1, dtype='f4'),
                      np.arange(-1, 1, 0.1, dtype='f8')]

        y_operands = [np.arange(1, 21, dtype='i4'),
                      np.arange(1, 21, dtype='i8'),
                      np.arange(1, 3, 0.1, dtype='f4'),
                      np.arange(1, 3, 0.1, dtype='f8')]

        for arraytype, x_operand, y_operand in zip(arraytypes, x_operands,
                                                    y_operands):
            for ufunc in ufunc_list:
                pyfunc = ufunc
                cr = compile_isolated(pyfunc, (arraytype, arraytype, arraytype))
                cfunc = cr.entry_point

                result = np.zeros(x_operand.size, dtype=x_operand.dtype)
                cfunc(x_operand, y_operand, result)
                control = np.zeros(x_operand.size, dtype=x_operand.dtype)
                ufunc(x_operand, y_operand, control)
                self.assertTrue(np.allclose(result, control))


if __name__ == '__main__':
    unittest.main()

