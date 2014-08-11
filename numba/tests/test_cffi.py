from __future__ import print_function, division, absolute_import

from numba import unittest_support as unittest
from numba import cffi_support, types
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
    """)
    C = ffi.dlopen(None)                     # loads the entire C namespace
    c_sin = C.sin


def use_cffi_sin(x):
    return c_sin(x) * 2


@unittest.skipUnless(cffi_support.SUPPORTED, "CFFI not supported")
class TestCFFI(TestCase):

    def test_cffi_sin_function(self, flags=enable_pyobj_flags):
        signature = cffi_support.map_type(ffi.typeof(c_sin))
        self.assertEqual(len(signature.args), 1)
        self.assertEqual(signature.args[0], types.double)

        pyfunc = use_cffi_sin
        cres = compile_isolated(pyfunc, [types.double], flags=flags)
        cfunc = cres.entry_point

        for x in [-1.2, -1, 0, 0.1]:
            self.assertPreciseEqual(pyfunc(x), cfunc(x))

    def test_cffi_sin_function_npm(self):
        self.test_cffi_sin_function(flags=no_pyobj_flags)


if __name__ == '__main__':
    unittest.main()
