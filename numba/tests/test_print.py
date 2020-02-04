import sys

import numpy as np

import unittest
from numba.core.compiler import compile_isolated, Flags
from numba import jit
from numba.core import types, errors, utils
from numba.tests.support import captured_stdout, tag, TestCase


enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

force_pyobj_flags = Flags()
force_pyobj_flags.set("force_pyobject")


def print_value(x):
    print(x)

def print_array_item(arr, i):
    print(arr[i].x)

def print_values(a, b, c):
    print(a, b, c)

def print_empty():
    print()

def print_string(x):
    print(x, "hop!", 3.5)

def print_vararg(a, b, c):
    print(a, b, *c)

def print_string_vararg(a, b, c):
    print(a, "hop!", b, *c)

def make_print_closure(x):
    def print_closure():
        return x
    return jit(nopython=True)(x)


class TestPrint(TestCase):

    def test_print_values(self):
        """
        Test printing a single argument value.
        """
        pyfunc = print_value

        def check_values(typ, values):
            cr = compile_isolated(pyfunc, (typ,))
            cfunc = cr.entry_point
            for val in values:
                with captured_stdout():
                    cfunc(val)
                    self.assertEqual(sys.stdout.getvalue(), str(val) + '\n')

        # Various scalars
        check_values(types.int32, (1, -234))
        check_values(types.int64, (1, -234,
                                   123456789876543210, -123456789876543210))
        check_values(types.uint64, (1, 234,
                                   123456789876543210, 2**63 + 123))
        check_values(types.boolean, (True, False))
        check_values(types.float64, (1.5, 100.0**10.0, float('nan')))
        check_values(types.complex64, (1+1j,))
        check_values(types.NPTimedelta('ms'), (np.timedelta64(100, 'ms'),))

        cr = compile_isolated(pyfunc, (types.float32,))
        cfunc = cr.entry_point
        with captured_stdout():
            cfunc(1.1)
            # Float32 will lose precision
            got = sys.stdout.getvalue()
            expect = '1.10000002384'
            self.assertTrue(got.startswith(expect))
            self.assertTrue(got.endswith('\n'))

        # NRT-enabled type
        with self.assertNoNRTLeak():
            x = [1, 3, 5, 7]
            with self.assertRefCount(x):
                check_values(types.List(types.int32), (x,))

        # Array will have to use object mode
        arraytype = types.Array(types.int32, 1, 'C')
        cr = compile_isolated(pyfunc, (arraytype,), flags=enable_pyobj_flags)
        cfunc = cr.entry_point
        with captured_stdout():
            cfunc(np.arange(10, dtype=np.int32))
            self.assertEqual(sys.stdout.getvalue(),
                             '[0 1 2 3 4 5 6 7 8 9]\n')

    def test_print_array_item(self):
        """
        Test printing a Numpy character sequence
        """
        dtype = np.dtype([('x', 'S4')])
        arr = np.frombuffer(bytearray(range(1, 9)), dtype=dtype)

        pyfunc = print_array_item
        cfunc = jit(nopython=True)(pyfunc)
        for i in range(len(arr)):
            with captured_stdout():
                cfunc(arr, i)
                self.assertEqual(sys.stdout.getvalue(), str(arr[i]['x']) + '\n')

    def test_print_multiple_values(self):
        pyfunc = print_values
        cr = compile_isolated(pyfunc, (types.int32,) * 3)
        cfunc = cr.entry_point
        with captured_stdout():
            cfunc(1, 2, 3)
            self.assertEqual(sys.stdout.getvalue(), '1 2 3\n')

    def test_print_nogil(self):
        pyfunc = print_values
        cfunc = jit(nopython=True, nogil=True)(pyfunc)
        with captured_stdout():
            cfunc(1, 2, 3)
            self.assertEqual(sys.stdout.getvalue(), '1 2 3\n')

    def test_print_empty(self):
        pyfunc = print_empty
        cr = compile_isolated(pyfunc, ())
        cfunc = cr.entry_point
        with captured_stdout():
            cfunc()
            self.assertEqual(sys.stdout.getvalue(), '\n')

    def test_print_strings(self):
        pyfunc = print_string
        cr = compile_isolated(pyfunc, (types.int32,))
        cfunc = cr.entry_point
        with captured_stdout():
            cfunc(1)
            self.assertEqual(sys.stdout.getvalue(), '1 hop! 3.5\n')

    def test_print_vararg(self):
        # Test *args support for print().  This is desired since
        # print() can use a dedicated IR node.
        pyfunc = print_vararg
        cfunc = jit(nopython=True)(pyfunc)
        with captured_stdout():
            cfunc(1, (2, 3), (4, 5j))
            self.assertEqual(sys.stdout.getvalue(), '1 (2, 3) 4 5j\n')

        pyfunc = print_string_vararg
        cfunc = jit(nopython=True)(pyfunc)
        with captured_stdout():
            cfunc(1, (2, 3), (4, 5j))
            self.assertEqual(sys.stdout.getvalue(), '1 hop! (2, 3) 4 5j\n')

    def test_inner_fn_print(self):
        @jit(nopython=True)
        def foo(x):
            print(x)

        @jit(nopython=True)
        def bar(x):
            foo(x)
            foo('hello')

        # Printing an array requires the Env.
        # We need to make sure the inner function can obtain the Env.
        x = np.arange(5)
        with captured_stdout():
            bar(x)
            self.assertEqual(sys.stdout.getvalue(), '[0 1 2 3 4]\nhello\n')

    def test_print_w_kwarg_raises(self):
        @jit(nopython=True)
        def print_kwarg():
            print('x', flush=True)

        with self.assertRaises(errors.UnsupportedError) as raises:
            print_kwarg()
        expected = ("Numba's print() function implementation does not support "
                    "keyword arguments.")
        self.assertIn(raises.exception.msg, expected)

    def test_print_no_truncation(self):
        ''' See: https://github.com/numba/numba/issues/3811
        '''
        @jit(nopython=True)
        def foo():
            print(''.join(['a'] * 10000))
        with captured_stdout():
            foo()
            self.assertEqual(sys.stdout.getvalue(), ''.join(['a'] * 10000) + '\n')

if __name__ == '__main__':
    unittest.main()
