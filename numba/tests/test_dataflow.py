from __future__ import print_function
import numba.unittest_support as unittest
from numba.compiler import compile_isolated, Flags
from numba import types


force_pyobj_flags = Flags()
force_pyobj_flags.set("force_pyobject")


def assignments(a):
    b = c = str(a)
    return b + c


def assignments2(a):
    b = c = d = str(a)
    return b + c + d


class TestDataFlow(unittest.TestCase):
    def test_assignments(self, flags=force_pyobj_flags):
        pyfunc = assignments
        cr = compile_isolated(pyfunc, (types.int32,), flags=flags)
        cfunc = cr.entry_point
        for x in [-1, 0, 1]:
            self.assertEqual(pyfunc(x), cfunc(x))

    def test_assignments2(self, flags=force_pyobj_flags):
        pyfunc = assignments2
        cr = compile_isolated(pyfunc, (types.int32,), flags=flags)
        cfunc = cr.entry_point
        for x in [-1, 0, 1]:
            self.assertEqual(pyfunc(x), cfunc(x))

        if flags is force_pyobj_flags:
            cfunc("a")


if __name__ == '__main__':
    unittest.main()

