from __future__ import print_function, division, absolute_import

from numba import unittest_support as unittest
from numba import jit, cffi_support, types
from numba.compiler import compile_isolated, Flags
from numba.tests.support import TestCase
from numba.tests.cffi_usecases import *

import numpy as np

enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

no_pyobj_flags = Flags()


@unittest.skipUnless(cffi_support.SUPPORTED, "CFFI not supported")
class TestCFFI(TestCase):

    def test_type_map(self):
        signature = cffi_support.map_type(ffi.typeof(cffi_sin))
        self.assertEqual(len(signature.args), 1)
        self.assertEqual(signature.args[0], types.double)

    def _test_function(self, pyfunc, flags=enable_pyobj_flags):
        cres = compile_isolated(pyfunc, [types.double], flags=flags)
        cfunc = cres.entry_point

        for x in [-1.2, -1, 0, 0.1, 3.14]:
            self.assertPreciseEqual(pyfunc(x), cfunc(x))

    def test_sin_function(self):
        self._test_function(use_cffi_sin)

    def test_sin_function_npm(self):
        self._test_function(use_cffi_sin, flags=no_pyobj_flags)

    def test_sin_function_ool(self, flags=enable_pyobj_flags):
        self._test_function(use_cffi_sin_ool)

    def test_sin_function_npm_ool(self):
        self._test_function(use_cffi_sin_ool, flags=no_pyobj_flags)

    def test_two_funcs(self):
        # Check that two constant functions don't get mixed up.
        self._test_function(use_two_funcs)

    def test_two_funcs_ool(self):
        self._test_function(use_two_funcs_ool)

    def test_function_pointer(self):
        pyfunc = use_func_pointer
        cfunc = jit(nopython=True)(pyfunc)
        for (fa, fb, x) in [
            (cffi_sin, cffi_cos, 1.0),
            (cffi_sin, cffi_cos, -1.0),
            (cffi_cos, cffi_sin, 1.0),
            (cffi_cos, cffi_sin, -1.0),
            (cffi_sin_ool, cffi_cos_ool, 1.0),
            (cffi_sin_ool, cffi_cos_ool, -1.0),
            (cffi_cos_ool, cffi_sin_ool, 1.0),
            (cffi_cos_ool, cffi_sin_ool, -1.0),
            (cffi_sin, cffi_cos_ool, 1.0),
            (cffi_sin, cffi_cos_ool, -1.0),
            (cffi_cos, cffi_sin_ool, 1.0),
            (cffi_cos, cffi_sin_ool, -1.0)]:
            expected = pyfunc(fa, fb, x)
            got = cfunc(fa, fb, x)
            self.assertEqual(got, expected)
        # A single specialization was compiled for all calls
        self.assertEqual(len(cfunc.overloads), 1, cfunc.overloads)

    def test_user_defined_sybols(self):
        pyfunc = use_user_defined_symbols
        cfunc = jit(nopython=True)(pyfunc)
        self.assertEqual(pyfunc(), cfunc())

    def _test_pass_numpy_array(self, pyfunc, dtype):
        x = np.arange(10).astype(dtype)
        cfunc = jit(nopython=True)(pyfunc)
        np.testing.assert_equal(pyfunc(x), cfunc(x))

    def test_pass_numpy_array_float32(self):
        self._test_pass_numpy_array(vector_sin_float32, np.float32)

    def test_pass_numpy_array_float64(self):
        self._test_pass_numpy_array(vector_sin_float64, np.float64)


if __name__ == '__main__':
    unittest.main()
