from __future__ import print_function, division, absolute_import

from numba import unittest_support as unittest
from numba import jit, cffi_support, types
from numba.compiler import compile_isolated, Flags
from .support import TestCase

enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

no_pyobj_flags = Flags()


if cffi_support.SUPPORTED:
    from cffi import FFI
    ffi = FFI()
    ffi.cdef("""
    double sin(double x);
    double cos(double x);
    """)
    C = ffi.dlopen(None)                     # loads the entire C namespace
    c_sin = C.sin
    c_cos = C.cos


def use_cffi_sin(x):
    return c_sin(x) * 2

def use_two_funcs(x):
    return c_sin(x) - c_cos(x)

def use_func_pointer(fa, fb, x):
    if x > 0:
        return fa(x)
    else:
        return fb(x)


@unittest.skipUnless(cffi_support.SUPPORTED, "CFFI not supported")
class TestCFFI(TestCase):

    def test_sin_function(self, flags=enable_pyobj_flags):
        signature = cffi_support.map_type(ffi.typeof(c_sin))
        self.assertEqual(len(signature.args), 1)
        self.assertEqual(signature.args[0], types.double)

        pyfunc = use_cffi_sin
        cres = compile_isolated(pyfunc, [types.double], flags=flags)
        cfunc = cres.entry_point

        for x in [-1.2, -1, 0, 0.1]:
            self.assertPreciseEqual(pyfunc(x), cfunc(x))

    def test_sin_function_npm(self):
        self.test_sin_function(flags=no_pyobj_flags)

    def test_two_funcs(self):
        # Check that two constant functions don't get mixed up.
        pyfunc = use_two_funcs
        cres = compile_isolated(pyfunc, [types.double])
        cfunc = cres.entry_point
        x = 3.14
        self.assertEqual(pyfunc(x), cfunc(x))

    def test_function_pointer(self):
        pyfunc = use_func_pointer
        cfunc = jit(nopython=True)(pyfunc)
        for (fa, fb, x) in [
            (c_sin, c_cos, 1.0),
            (c_sin, c_cos, -1.0),
            (c_cos, c_sin, 1.0),
            (c_cos, c_sin, -1.0)]:
            expected = pyfunc(fa, fb, x)
            got = cfunc(fa, fb, x)
            self.assertEqual(got, expected)
        # A single specialization was compiled for all calls
        self.assertEqual(len(cfunc.overloads), 1, cfunc.overloads)


if __name__ == '__main__':
    unittest.main()
