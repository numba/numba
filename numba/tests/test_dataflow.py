from __future__ import print_function

import numba.unittest_support as unittest
from numba.compiler import compile_isolated, Flags
from numba import types
from .support import TestCase


enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

force_pyobj_flags = Flags()
force_pyobj_flags.set("force_pyobject")

no_pyobj_flags = Flags()


def assignments(a):
    b = c = str(a)
    return b + c


def assignments2(a):
    b = c = d = str(a)
    return b + c + d


# Use cases for issue #503

def var_propagate1(a, b):
    c = (a if a > b else b) + 5
    return c


def var_propagate2(a, b):
    c = 5 + (a if a > b else b + 12) / 2.0
    return c


class TestDataFlow(TestCase):

    def test_assignments(self, flags=force_pyobj_flags):
        pyfunc = assignments
        cr = compile_isolated(pyfunc, (types.int32,), flags=flags)
        cfunc = cr.entry_point
        for x in [-1, 0, 1]:
            self.assertPreciseEqual(pyfunc(x), cfunc(x))

    def test_assignments2(self, flags=force_pyobj_flags):
        pyfunc = assignments2
        cr = compile_isolated(pyfunc, (types.int32,), flags=flags)
        cfunc = cr.entry_point
        for x in [-1, 0, 1]:
            self.assertPreciseEqual(pyfunc(x), cfunc(x))

        if flags is force_pyobj_flags:
            cfunc("a")

    def test_var_propagate1(self):
        # The dataflow analysis must be good enough for native mode
        # compilation to succeed, hence the no_pyobj_flags.
        pyfunc = var_propagate1
        cr = compile_isolated(pyfunc, (types.int32, types.int32),
                              flags=no_pyobj_flags)
        cfunc = cr.entry_point
        args = (2, 3)
        self.assertPreciseEqual(cfunc(*args), pyfunc(*args))

    def test_var_propagate2(self):
        # The dataflow analysis must be good enough for native mode
        # compilation to succeed, hence the no_pyobj_flags.
        pyfunc = var_propagate2
        cr = compile_isolated(pyfunc, (types.int32, types.int32),
                              flags=no_pyobj_flags)
        cfunc = cr.entry_point
        args = (2, 3)
        self.assertPreciseEqual(cfunc(*args), pyfunc(*args))


if __name__ == '__main__':
    unittest.main()

