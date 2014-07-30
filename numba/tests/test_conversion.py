from __future__ import print_function
import itertools
import numba.unittest_support as unittest
from numba.compiler import compile_isolated, Flags
from numba import types, jit, numpy_support

import numpy as np
import sys

def identity(x):
    return x

def addition(x, y):
    return x + y

def equality(x, y):
    return x == y

def foobar(x, y, z):
    return x

class TestConversion(unittest.TestCase):
    """
    Testing Python to Native conversion
    """
    def test_complex_identity(self):
        pyfunc = identity
        cres = compile_isolated(pyfunc, [types.complex64],
                                return_type=types.complex64)

        xs = [1.0j, (1+1j), (-1-1j), (1+0j)]
        for x in xs:
            self.assertEqual(cres.entry_point(x=x), x)
        for x in np.complex64(xs):
            self.assertEqual(cres.entry_point(x=x), x)

        cres = compile_isolated(pyfunc, [types.complex128],
                                return_type=types.complex128)

        xs = [1.0j, (1+1j), (-1-1j), (1+0j)]
        for x in xs:
            self.assertEqual(cres.entry_point(x=x), x)
        for x in np.complex128(xs):
            self.assertEqual(cres.entry_point(x=x), x)

    def test_complex_addition(self):
        pyfunc = addition
        cres = compile_isolated(pyfunc, [types.complex64, types.complex64],
                                return_type=types.complex64)

        xs = [1.0j, (1+1j), (-1-1j), (1+0j)]
        for x in xs:
            y = x
            self.assertEqual(cres.entry_point(x, y), x + y)
        for x in np.complex64(xs):
            y = x
            self.assertEqual(cres.entry_point(x, y), x + y)


        cres = compile_isolated(pyfunc, [types.complex128, types.complex128],
                                return_type=types.complex128)

        xs = [1.0j, (1+1j), (-1-1j), (1+0j)]
        for x in xs:
            y = x
            self.assertEqual(cres.entry_point(x, y), x + y)
        for x in np.complex128(xs):
            y = x
            self.assertEqual(cres.entry_point(x, y), x + y)

    def test_boolean_as_int(self):
        pyfunc = equality
        cres = compile_isolated(pyfunc, [types.boolean, types.intp])
        cfunc = cres.entry_point

        xs = True, False
        ys = -1, 0, 1

        for xs, ys in itertools.product(xs, ys):
            self.assertEqual(pyfunc(xs, ys), cfunc(xs, ys))

    def test_boolean_as_float(self):
        pyfunc = equality
        cres = compile_isolated(pyfunc, [types.boolean, types.float64])
        cfunc = cres.entry_point

        xs = True, False
        ys = -1, 0, 1

        for xs, ys in itertools.product(xs, ys):
            self.assertEqual(pyfunc(xs, ys), cfunc(xs, ys))

    def test_boolean_eq_boolean(self):
        pyfunc = equality
        cres = compile_isolated(pyfunc, [types.boolean, types.boolean])
        cfunc = cres.entry_point

        xs = True, False
        ys = True, False

        for xs, ys in itertools.product(xs, ys):
            self.assertEqual(pyfunc(xs, ys), cfunc(xs, ys))

    # test when a function parameters are jitted as unsigned types
    # the function is called with negative parameters the Python error 
    # that it generates is correctly handled -- a Python error is returned to the user
    # For more info, see the comment in Include/longobject.h for _PyArray_AsByteArray 
    # which PyLong_AsUnsignedLongLong calls
    def test_negative_to_unsigned(self):
        def f(x):
            return x
        # TypeError is for 2.6
        if sys.version_info >= (2, 7):
            with self.assertRaises(OverflowError):
                jit('uintp(uintp)', nopython=True)(f)(-5)
        else:
            with self.assertRaises(TypeError):
                jit('uintp(uintp)', nopython=True)(f)(-5)

    # test the switch logic in callwraper.py:build_wrapper() works for more than one argument
    # and where the error occurs 
    def test_multiple_args_negative_to_unsigned(self): 
        pyfunc = foobar
        cres = compile_isolated(pyfunc, [types.uint64, types.uint64, types.uint64],
                                return_type=types.uint64)
        cfunc = cres.entry_point
        test_fail_args = ((-1, 0, 1), (0, -1, 1), (0, 1, -1))
        # TypeError is for 2.6
        if sys.version_info >= (2, 7):
            with self.assertRaises(OverflowError):
                for a, b, c in test_fail_args:
                    cfunc(a, b, c) 
        else:
            with self.assertRaises(TypeError):
                for a, b, c in test_fail_args:
                    cfunc(a, b, c) 

    # test switch logic of callwraper.py:build_wrapper() with records as function parameters
    def test_multiple_args_records(self): 
        pyfunc = foobar

        mystruct_dt = np.dtype([('p', np.float64),
                           ('row', np.float64),
                           ('col', np.float64)])
        mystruct = numpy_support.from_dtype(mystruct_dt)

        cres = compile_isolated(pyfunc, [mystruct[:], types.uint64, types.uint64],
                return_type=mystruct[:])
        cfunc = cres.entry_point

        st1 = np.recarray(3, dtype=mystruct_dt)
        st2 = np.recarray(3, dtype=mystruct_dt)

        st1.p = np.arange(st1.size) + 1
        st1.row = np.arange(st1.size) + 1
        st1.col = np.arange(st1.size) + 1

        st2.p = np.arange(st2.size) + 1
        st2.row = np.arange(st2.size) + 1
        st2.col = np.arange(st2.size) + 1

        test_fail_args = ((st1, -1, st2), (st1, st2, -1))
        
        # TypeError is for 2.6
        if sys.version_info >= (2, 7):
            with self.assertRaises(OverflowError):
                for a, b, c in test_fail_args:
                    cfunc(a, b, c) 
        else:
            with self.assertRaises(TypeError):
                for a, b, c in test_fail_args:
                    cfunc(a, b, c) 

    # test switch logic of callwraper.py:build_wrapper() with no function parameters
    def test_with_no_parameters(self):
        def f():
            pass 
        self.assertEqual(f(), jit('()', nopython=True)(f)())

if __name__ == '__main__':
    unittest.main()
