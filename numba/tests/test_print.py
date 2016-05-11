from __future__ import print_function

import sys

import numpy as np

import numba.unittest_support as unittest
from numba.compiler import compile_isolated, Flags
from numba import types
from .support import captured_stdout, tag


enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

force_pyobj_flags = Flags()
force_pyobj_flags.set("force_pyobject")


def print_value(x):
    print(x)

def print_values(a, b, c):
    print(a, b, c)

def print_empty():
    print()


class TestPrint(unittest.TestCase):

    @tag('important')
    def test_print(self):
        pyfunc = print_value

        def check_values(typ, values):
            cr = compile_isolated(pyfunc, (typ,))
            cfunc = cr.entry_point
            for val in values:
                with captured_stdout():
                    cfunc(val)
                    self.assertEqual(sys.stdout.getvalue(), str(val) + '\n')

        check_values(types.int32, (1, -234))
        check_values(types.int64, (1, -234,
                                   123456789876543210, -123456789876543210))
        check_values(types.uint64, (1, 234,
                                   123456789876543210, 2**63 + 123))
        check_values(types.boolean, (True, False))

        cr = compile_isolated(pyfunc, (types.float32,))
        cfunc = cr.entry_point
        with captured_stdout():
            cfunc(1.1)
            # Float32 will lose precision
            got = sys.stdout.getvalue()
            expect = '1.10000002384'
            self.assertTrue(got.startswith(expect))
            self.assertTrue(got.endswith('\n'))

        cr = compile_isolated(pyfunc, (types.float64,))
        cfunc = cr.entry_point
        with captured_stdout():
            cfunc(100.0**10.0)
            self.assertEqual(sys.stdout.getvalue(), '1e+20\n')

        # Array will have to use object mode
        arraytype = types.Array(types.int32, 1, 'C')
        cr = compile_isolated(pyfunc, (arraytype,), flags=enable_pyobj_flags)
        cfunc = cr.entry_point
        with captured_stdout():
            cfunc(np.arange(10))
            self.assertEqual(sys.stdout.getvalue(),
                             '[0 1 2 3 4 5 6 7 8 9]\n')

    @tag('important')
    def test_print_multiple_values(self):
        pyfunc = print_values
        cr = compile_isolated(pyfunc, (types.int32,) * 3)
        cfunc = cr.entry_point
        with captured_stdout():
            cfunc(1, 2, 3)
            self.assertEqual(sys.stdout.getvalue(), '1 2 3\n')

    @tag('important')
    def test_print_empty(self):
        pyfunc = print_empty
        cr = compile_isolated(pyfunc, ())
        cfunc = cr.entry_point
        with captured_stdout():
            cfunc()
            self.assertEqual(sys.stdout.getvalue(), '\n')


if __name__ == '__main__':
    unittest.main()
