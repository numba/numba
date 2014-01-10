from __future__ import print_function
import unittest
from numba.compiler import compile_isolated, Flags
from numba import types, utils
from numba.tests import usecases
import numpy as np
import sys

enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

force_pyobj_flags = Flags()
force_pyobj_flags.set("force_pyobject")


def print_value(x):
    print(x)

class TestPrint(unittest.TestCase):

    def test_print(self):
        pyfunc = print_value

        cr = compile_isolated(pyfunc, (types.int32,))
        cfunc = cr.entry_point
        cfunc(1)
        self.assertEqual(sys.stdout.getvalue().strip(), '1')

        cr = compile_isolated(pyfunc, (types.int64,))
        cfunc = cr.entry_point
        cfunc(1)
        self.assertEqual(sys.stdout.getvalue().strip(), '1')

        cr = compile_isolated(pyfunc, (types.float32,))
        cfunc = cr.entry_point
        cfunc(1.1)
        self.assertEqual(sys.stdout.getvalue().strip(), '1.1')

        cr = compile_isolated(pyfunc, (types.float64,))
        cfunc = cr.entry_point
        cfunc(100.0**10.0)
        self.assertEqual(sys.stdout.getvalue().strip(), '1e+20')

        arraytype = types.Array(types.int32, 1, 'C')
        cr = compile_isolated(pyfunc, (arraytype,))
        cfunc = cr.entry_point
        cfunc(np.arange(10))
        self.assertEqual(sys.stdout.getvalue().strip(),'[0 1 2 3 4 5 6 7 8 9]')


if __name__ == '__main__':
    unittest.main(buffer=True)

