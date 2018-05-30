"""
Tests scipy specific features
"""
from __future__ import print_function, absolute_import


import ctypes

try:
    import scipy
    has_scipy = True
except ImportError:
    has_scipy = False

try:
    import cffi
    has_cffi = True
except ImportError:
    has_cffi = False

import numba.unittest_support as unittest
from .support import TestCase
from numba import njit, types, typeof


skip_if_no_scipy = unittest.skipIf(not has_scipy, "Skip, no scipy")
skip_if_no_cffi = unittest.skipIf(not has_cffi, "Skip, no cffi")


@skip_if_no_scipy
class TestLowLevelCallable(TestCase):

    def make_ctypes_llc(self):
        cproto = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)

        @cproto
        def twice(x):
            return x + x

        llc = scipy.LowLevelCallable(twice)
        return llc

    def make_cffi_llc(self):
        ffi = cffi.FFI()
        self.ffi = ffi

        @ffi.callback('int(int, int)')
        def twice(x):
            return x + x

        llc = scipy.LowLevelCallable(twice)
        return llc

    def test_typeof_ctypes(self):
        llc = self.make_ctypes_llc()
        llc_type = typeof(llc)
        self.assertIsInstance(llc_type, types.LowLevelCallable)

    def test_typeof_cffi(self):
        llc = self.make_cffi_llc()
        llc_type = typeof(llc)
        self.assertIsInstance(llc_type, types.LowLevelCallable)

    def test_call_ctypes(self):
        @njit
        def caller(fn, x):
            return fn.function(x)

        llc = self.make_ctypes_llc()
        got = caller(llc, 123)
        expect = llc.function(123)
        self.assertEqual(got, expect)

    def test_call_cffi(self):
        @njit
        def caller(fn, x):
            return fn.function(x)

        llc = self.make_ctypes_llc()
        got = caller(llc, 123)
        expect = llc.function(123)
        self.assertEqual(got, expect)


if __name__ == "__main__":
    unittest.main()
