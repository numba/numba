from __future__ import print_function
import numba.unittest_support as unittest
from numba.compiler import compile_isolated, Flags
from numba import types

enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

no_pyobj_flags = Flags()


def isnan(x):
    return x != x

def isequal(x):
    return x == x


class TestNaN(unittest.TestCase):

    def test_nans(self, flags=enable_pyobj_flags):
        pyfunc = isnan
        cr = compile_isolated(pyfunc, (types.float64,), flags=flags)
        cfunc = cr.entry_point

        self.assertTrue(cfunc(float('nan')))
        self.assertFalse(cfunc(1.0))

        pyfunc = isequal
        cr = compile_isolated(pyfunc, (types.float64,), flags=flags)
        cfunc = cr.entry_point

        self.assertFalse(cfunc(float('nan')))
        self.assertTrue(cfunc(1.0))

    def test_nans_npm(self):
        self.test_nans(flags=no_pyobj_flags)

if __name__ == '__main__':
    unittest.main()

