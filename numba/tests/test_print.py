from __future__ import print_function
import numba.unittest_support as unittest
from contextlib import contextmanager
from numba.compiler import compile_isolated, Flags
from numba import types
from numba.io_support import StringIO
import numpy as np
import sys

enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

force_pyobj_flags = Flags()
force_pyobj_flags.set("force_pyobject")


def print_value(x):
    print(x)


@contextmanager
def swap_stdout():
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    yield
    sys.stdout = old_stdout


class TestPrint(unittest.TestCase):
    def test_print(self):
        pyfunc = print_value

        cr = compile_isolated(pyfunc, (types.int32,))
        cfunc = cr.entry_point

        with swap_stdout():
            cfunc(1)
            self.assertEqual(sys.stdout.getvalue().strip(), '1')

        cr = compile_isolated(pyfunc, (types.int64,))
        cfunc = cr.entry_point
        with swap_stdout():
            cfunc(1)
            self.assertEqual(sys.stdout.getvalue().strip(), '1')

        cr = compile_isolated(pyfunc, (types.float32,))
        cfunc = cr.entry_point
        with swap_stdout():
            cfunc(1.1)
            # Float32 will loose precision
            got = sys.stdout.getvalue().strip()
            expect = '1.10000002384'
            self.assertTrue(got.startswith(expect))

        cr = compile_isolated(pyfunc, (types.float64,))
        cfunc = cr.entry_point
        with swap_stdout():
            cfunc(100.0**10.0)
            self.assertEqual(sys.stdout.getvalue().strip(), '1e+20')

        # Array will have to use object mode
        arraytype = types.Array(types.int32, 1, 'C')
        cr = compile_isolated(pyfunc, (arraytype,), flags=enable_pyobj_flags)
        cfunc = cr.entry_point
        with swap_stdout():
            cfunc(np.arange(10))
            self.assertEqual(sys.stdout.getvalue().strip(),
                             '[0 1 2 3 4 5 6 7 8 9]')


if __name__ == '__main__':
    unittest.main(buffer=True)

