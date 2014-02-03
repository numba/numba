from __future__ import print_function, division, absolute_import
from numba import unittest_support as unittest
from numba import cffi_support, types
from numba.compiler import compile_isolated


if cffi_support.SUPPORTED:
    from cffi import FFI
    ffi = FFI()
    ffi.cdef("""
    double sin(double x);
    """)
    C = ffi.dlopen(None)                     # loads the entire C namespace
    c_sin = C.sin


def use_cffi_sin(x):
    return c_sin(x)


@unittest.skipIf(not cffi_support.SUPPORTED, "CFFI not supported")
class TestCFFI(unittest.TestCase):
    def test_cffi_sin_function(self):
        signature = cffi_support.map_type(ffi.typeof(c_sin))
        self.assertEqual(len(signature.args), 1)
        self.assertEqual(signature.args[0], types.double)

        pyfunc = use_cffi_sin
        cres = compile_isolated(pyfunc, [types.double])
        cfunc = cres.entry_point

        for x in [-1.2, -1, 0, 0.1]:
            self.assertEqual(pyfunc(x), cfunc(x))


if __name__ == '__main__':
    unittest.main()
