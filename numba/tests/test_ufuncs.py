from __future__ import print_function
import numba.unittest_support as unittest
import numpy as np
from numba.compiler import compile_isolated, Flags
from numba import types, utils
from numba.tests import usecases

enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

no_pyobj_flags = Flags()


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

    def test_unary_ufuncs(self, flags=enable_pyobj_flags):
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
                cr = compile_isolated(pyfunc, (arraytype, arraytype),
                                      flags=flags)
                cfunc = cr.entry_point

                result = np.zeros(x_operand.size, dtype=x_operand.dtype)
                cfunc(x_operand, result)
                control = np.zeros(x_operand.size, dtype=x_operand.dtype)
                ufunc(x_operand, control)
                self.assertTrue(np.allclose(result, control))

    def test_unary_ufuncs_npm(self):
        self.test_unary_ufuncs(flags=no_pyobj_flags)

    def test_binary_ufuncs(self, flags=enable_pyobj_flags):
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
                cr = compile_isolated(pyfunc,
                                      (arraytype, arraytype, arraytype),
                                      flags=flags)
                cfunc = cr.entry_point

                result = np.zeros(xy_operand.size, dtype=xy_operand.dtype)
                cfunc(xy_operand, xy_operand, result)
                control = np.zeros(xy_operand.size, dtype=xy_operand.dtype)
                ufunc(xy_operand, xy_operand, control)
                self.assertTrue(np.allclose(result, control))

    def test_binary_ufuncs_npm(self):
        self.test_binary_ufuncs(flags=no_pyobj_flags)

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

        print(utils.benchmark(bm_python, maxsec=.1))
        print(utils.benchmark(bm_numba, maxsec=.1))
        assert np.allclose(control, result)

    def test_divide_ufuncs(self, flags=enable_pyobj_flags):
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
                cr = compile_isolated(pyfunc,
                                      (arraytype, arraytype, arraytype),
                                      flags=flags)
                cfunc = cr.entry_point

                result = np.zeros(x_operand.size, dtype=x_operand.dtype)
                cfunc(x_operand, y_operand, result)
                control = np.zeros(x_operand.size, dtype=x_operand.dtype)
                ufunc(x_operand, y_operand, control)
                self.assertTrue(np.allclose(result, control))

        # NumPy integer division should return zero by default
        arraytype = types.Array(types.int32, 1, 'C')
        x_operand = np.array([0], dtype='i4')
        y_operand = np.array([0], dtype='i4')
        for ufunc in ufunc_list:
            pyfunc = ufunc
            cr = compile_isolated(pyfunc,
                                  (arraytype, arraytype, arraytype),
                                  flags=flags)
            cfunc = cr.entry_point

            # Initialize result value to something other than zero
            result = np.array([999], dtype='i4')
            cfunc(x_operand, y_operand, result)
            self.assertTrue(np.allclose(result, np.array([0], dtype='i4')))

        # NumPy float division should return NaN or inf by default
        arraytype = types.Array(types.float32, 1, 'C')
        x_operand = np.array([0.0, 1.0], dtype='f4')
        y_operand = np.array([0.0, 0.0], dtype='f4')
        for ufunc in ufunc_list:
            pyfunc = ufunc
            cr = compile_isolated(pyfunc,
                                  (arraytype, arraytype, arraytype),
                                  flags=flags)
            cfunc = cr.entry_point

            # Initialize result value to something other than zero
            result = np.array([999.9, 999.9], dtype='f4')
            cfunc(x_operand, y_operand, result)
            self.assertTrue(np.isnan(result[0]))
            self.assertTrue(result[1] == np.inf)

    def test_divide_ufuncs_npm(self):
        self.test_divide_ufuncs(flags=no_pyobj_flags)


if __name__ == '__main__':
    unittest.main()

