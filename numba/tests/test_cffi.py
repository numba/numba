from __future__ import print_function, division, absolute_import

import array
import numpy as np
import sys

from numba import unittest_support as unittest
from numba import jit, cffi_support, types, errors
from numba.compiler import compile_isolated, Flags
from numba.tests.support import TestCase, tag

import numba.tests.cffi_usecases as mod


enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

no_pyobj_flags = Flags()


@unittest.skipUnless(cffi_support.SUPPORTED,
                     "CFFI not supported -- please install the cffi module")
class TestCFFI(TestCase):

    # Need to run the tests serially because of race conditions in
    # cffi's OOL mode.
    _numba_parallel_test_ = False

    def setUp(self):
        mod.init()
        mod.init_ool()

    def test_type_map(self):
        signature = cffi_support.map_type(mod.ffi.typeof(mod.cffi_sin))
        self.assertEqual(len(signature.args), 1)
        self.assertEqual(signature.args[0], types.double)

    def _test_function(self, pyfunc, flags=enable_pyobj_flags):
        cres = compile_isolated(pyfunc, [types.double], flags=flags)
        cfunc = cres.entry_point

        for x in [-1.2, -1, 0, 0.1, 3.14]:
            self.assertPreciseEqual(pyfunc(x), cfunc(x))

    def test_sin_function(self):
        self._test_function(mod.use_cffi_sin)

    @tag('important')
    def test_sin_function_npm(self):
        self._test_function(mod.use_cffi_sin, flags=no_pyobj_flags)

    def test_sin_function_ool(self, flags=enable_pyobj_flags):
        self._test_function(mod.use_cffi_sin_ool)

    def test_sin_function_npm_ool(self):
        self._test_function(mod.use_cffi_sin_ool, flags=no_pyobj_flags)

    def test_two_funcs(self):
        # Check that two constant functions don't get mixed up.
        self._test_function(mod.use_two_funcs)

    def test_two_funcs_ool(self):
        self._test_function(mod.use_two_funcs_ool)

    def test_function_pointer(self):
        pyfunc = mod.use_func_pointer
        cfunc = jit(nopython=True)(pyfunc)
        for (fa, fb, x) in [
            (mod.cffi_sin, mod.cffi_cos, 1.0),
            (mod.cffi_sin, mod.cffi_cos, -1.0),
            (mod.cffi_cos, mod.cffi_sin, 1.0),
            (mod.cffi_cos, mod.cffi_sin, -1.0),
            (mod.cffi_sin_ool, mod.cffi_cos_ool, 1.0),
            (mod.cffi_sin_ool, mod.cffi_cos_ool, -1.0),
            (mod.cffi_cos_ool, mod.cffi_sin_ool, 1.0),
            (mod.cffi_cos_ool, mod.cffi_sin_ool, -1.0),
            (mod.cffi_sin, mod.cffi_cos_ool, 1.0),
            (mod.cffi_sin, mod.cffi_cos_ool, -1.0),
            (mod.cffi_cos, mod.cffi_sin_ool, 1.0),
            (mod.cffi_cos, mod.cffi_sin_ool, -1.0)]:
            expected = pyfunc(fa, fb, x)
            got = cfunc(fa, fb, x)
            self.assertEqual(got, expected)
        # A single specialization was compiled for all calls
        self.assertEqual(len(cfunc.overloads), 1, cfunc.overloads)

    def test_user_defined_symbols(self):
        pyfunc = mod.use_user_defined_symbols
        cfunc = jit(nopython=True)(pyfunc)
        self.assertEqual(pyfunc(), cfunc())

    def check_vector_sin(self, cfunc, x, y):
        cfunc(x, y)
        np.testing.assert_allclose(y, np.sin(x))

    def _test_from_buffer_numpy_array(self, pyfunc, dtype):
        x = np.arange(10).astype(dtype)
        y = np.zeros_like(x)
        cfunc = jit(nopython=True)(pyfunc)
        self.check_vector_sin(cfunc, x, y)

    @tag('important')
    def test_from_buffer_float32(self):
        self._test_from_buffer_numpy_array(mod.vector_sin_float32, np.float32)

    def test_from_buffer_float64(self):
        self._test_from_buffer_numpy_array(mod.vector_sin_float64, np.float64)

    def test_from_buffer_struct(self):
        n = 10
        x = np.arange(n) + np.arange(n * 2, n * 3) * 1j
        y = np.zeros(n)
        real_cfunc = jit(nopython=True)(mod.vector_extract_real)
        real_cfunc(x, y)
        np.testing.assert_equal(x.real, y)
        imag_cfunc = jit(nopython=True)(mod.vector_extract_imag)
        imag_cfunc(x, y)
        np.testing.assert_equal(x.imag, y)


    @unittest.skipIf(sys.version_info < (3,),
                     "buffer protocol on array.array needs Python 3+")
    def test_from_buffer_pyarray(self):
        pyfunc = mod.vector_sin_float32
        cfunc = jit(nopython=True)(pyfunc)
        x = array.array("f", range(10))
        y = array.array("f", [0] * len(x))
        self.check_vector_sin(cfunc, x, y)

    def test_from_buffer_error(self):
        pyfunc = mod.vector_sin_float32
        cfunc = jit(nopython=True)(pyfunc)
        # Non-contiguous array
        x = np.arange(10).astype(np.float32)[::2]
        y = np.zeros_like(x)
        with self.assertRaises(errors.TypingError) as raises:
            cfunc(x, y)
        self.assertIn("from_buffer() unsupported on non-contiguous buffers",
                      str(raises.exception))


if __name__ == '__main__':
    unittest.main()
