"""
Tests for @cfunc and friends.
"""

from __future__ import division, print_function, absolute_import

import ctypes
import os
import subprocess
import sys

import numpy as np

from numba import unittest_support as unittest
from numba import cfunc, carray, farray, types, typing, utils
from numba import cffi_support
from .support import TestCase, tag, captured_stderr
from .test_dispatcher import BaseCacheTest


def add_usecase(a, b):
    return a + b

def div_usecase(a, b):
    c = a / b
    return c

def square_usecase(a):
    return a ** 2

add_sig = "float64(float64, float64)"

div_sig = "float64(int64, int64)"

square_sig = "float64(float64)"

def objmode_usecase(a, b):
    object()
    return a + b

# Test functions for carray() and farray()

CARRAY_USECASE_OUT_LEN = 8

def make_cfarray_usecase(func):

    def cfarray_usecase(in_ptr, out_ptr, m, n):
        # Tuple shape
        in_ = func(in_ptr, (m, n))
        # Integer shape
        out = func(out_ptr, CARRAY_USECASE_OUT_LEN)
        out[0] = in_.ndim
        out[1:3] = in_.shape
        out[3:5] = in_.strides
        out[5] = in_.flags.c_contiguous
        out[6] = in_.flags.f_contiguous
        s = 0
        for i, j in np.ndindex(m, n):
            s += in_[i, j] * (i - j)
        out[7] = s

    return cfarray_usecase

carray_usecase = make_cfarray_usecase(carray)
farray_usecase = make_cfarray_usecase(farray)


def make_cfarray_dtype_usecase(func):
    # Same as make_cfarray_usecase(), but with explicit dtype.

    def cfarray_usecase(in_ptr, out_ptr, m, n):
        # Tuple shape
        in_ = func(in_ptr, (m, n), dtype=np.float32)
        # Integer shape
        out = func(out_ptr, CARRAY_USECASE_OUT_LEN, np.float32)
        out[0] = in_.ndim
        out[1:3] = in_.shape
        out[3:5] = in_.strides
        out[5] = in_.flags.c_contiguous
        out[6] = in_.flags.f_contiguous
        s = 0
        for i, j in np.ndindex(m, n):
            s += in_[i, j] * (i - j)
        out[7] = s

    return cfarray_usecase

carray_dtype_usecase = make_cfarray_dtype_usecase(carray)
farray_dtype_usecase = make_cfarray_dtype_usecase(farray)

carray_float32_usecase_sig = types.void(types.CPointer(types.float32),
                                        types.CPointer(types.float32),
                                        types.intp, types.intp)

carray_float64_usecase_sig = types.void(types.CPointer(types.float64),
                                        types.CPointer(types.float64),
                                        types.intp, types.intp)

carray_voidptr_usecase_sig = types.void(types.voidptr, types.voidptr,
                                        types.intp, types.intp)


class TestCFunc(TestCase):

    @tag('important')
    def test_basic(self):
        """
        Basic usage and properties of a cfunc.
        """
        f = cfunc(add_sig)(add_usecase)

        self.assertEqual(f.__name__, "add_usecase")
        self.assertEqual(f.__qualname__, "add_usecase")
        self.assertIs(f.__wrapped__, add_usecase)

        symbol = f.native_name
        self.assertIsInstance(symbol, str)
        self.assertIn("add_usecase", symbol)

        addr = f.address
        self.assertIsInstance(addr, utils.INT_TYPES)

        ct = f.ctypes
        self.assertEqual(ctypes.cast(ct, ctypes.c_void_p).value, addr)

        self.assertPreciseEqual(ct(2.0, 3.5), 5.5)

    @tag('important')
    @unittest.skipUnless(cffi_support.SUPPORTED,
                         "CFFI not supported -- please install the cffi module")
    def test_cffi(self):
        from . import cffi_usecases
        ffi, lib = cffi_usecases.load_inline_module()

        f = cfunc(square_sig)(square_usecase)

        res = lib._numba_test_funcptr(f.cffi)
        self.assertPreciseEqual(res, 2.25)  # 1.5 ** 2

    def test_locals(self):
        # By forcing the intermediate result into an integer, we
        # truncate the ultimate function result
        f = cfunc(div_sig, locals={'c': types.int64})(div_usecase)
        self.assertPreciseEqual(f.ctypes(8, 3), 2.0)

    @tag('important')
    def test_errors(self):
        f = cfunc(div_sig)(div_usecase)

        with captured_stderr() as err:
            self.assertPreciseEqual(f.ctypes(5, 2), 2.5)
        self.assertEqual(err.getvalue(), "")

        with captured_stderr() as err:
            res = f.ctypes(5, 0)
            # This is just a side effect of Numba zero-initializing
            # stack variables, and could change in the future.
            self.assertPreciseEqual(res, 0.0)
        err = err.getvalue()
        self.assertIn("ZeroDivisionError: (\'division by zero\',)", err)
        if sys.version_info >= (3,):
            self.assertIn("Exception ignored", err)
        else:
            self.assertIn(" ignored", err)

    def test_llvm_ir(self):
        f = cfunc(add_sig)(add_usecase)
        ir = f.inspect_llvm()
        self.assertIn(f.native_name, ir)
        self.assertIn("fadd double", ir)

    def test_object_mode(self):
        """
        Object mode is currently unsupported.
        """
        with self.assertRaises(NotImplementedError):
            cfunc(add_sig, forceobj=True)(add_usecase)
        with self.assertTypingError() as raises:
            cfunc(add_sig)(objmode_usecase)
        self.assertIn("Untyped global name 'object'", str(raises.exception))


class TestCFuncCache(BaseCacheTest):

    here = os.path.dirname(__file__)
    usecases_file = os.path.join(here, "cfunc_cache_usecases.py")
    modname = "cfunc_caching_test_fodder"

    def run_in_separate_process(self):
        # Cached functions can be run from a distinct process.
        code = """if 1:
            import sys

            sys.path.insert(0, %(tempdir)r)
            mod = __import__(%(modname)r)
            mod.self_test()

            f = mod.add_usecase
            assert f.cache_hits == 1
            f = mod.outer
            assert f.cache_hits == 1
            f = mod.div_usecase
            assert f.cache_hits == 1
            """ % dict(tempdir=self.tempdir, modname=self.modname)

        popen = subprocess.Popen([sys.executable, "-c", code],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = popen.communicate()
        if popen.returncode != 0:
            raise AssertionError("process failed with code %s: stderr follows\n%s\n"
                                 % (popen.returncode, err.decode()))

    def check_module(self, mod):
        mod.self_test()

    @tag('important')
    def test_caching(self):
        self.check_pycache(0)
        mod = self.import_module()
        self.check_pycache(6)  # 3 index, 3 data

        self.assertEqual(mod.add_usecase.cache_hits, 0)
        self.assertEqual(mod.outer.cache_hits, 0)
        self.assertEqual(mod.add_nocache_usecase.cache_hits, 0)
        self.assertEqual(mod.div_usecase.cache_hits, 0)
        self.check_module(mod)

        # Reload module to hit the cache
        mod = self.import_module()
        self.check_pycache(6)  # 3 index, 3 data

        self.assertEqual(mod.add_usecase.cache_hits, 1)
        self.assertEqual(mod.outer.cache_hits, 1)
        self.assertEqual(mod.add_nocache_usecase.cache_hits, 0)
        self.assertEqual(mod.div_usecase.cache_hits, 1)
        self.check_module(mod)

        self.run_in_separate_process()


class TestCArray(TestCase):
    """
    Tests for carray() and farray().
    """

    def run_carray_usecase(self, pointer_factory, func):
        a = np.arange(10, 16).reshape((2, 3)).astype(np.float32)
        out = np.empty(CARRAY_USECASE_OUT_LEN, dtype=np.float32)
        func(pointer_factory(a), pointer_factory(out), *a.shape)
        return out

    def check_carray_usecase(self, pointer_factory, pyfunc, cfunc):
        expected = self.run_carray_usecase(pointer_factory, pyfunc)
        got = self.run_carray_usecase(pointer_factory, cfunc)
        self.assertPreciseEqual(expected, got)

    def make_voidptr(self, arr):
        return arr.ctypes.data_as(ctypes.c_void_p)

    def make_float32_pointer(self, arr):
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def make_float64_pointer(self, arr):
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def check_carray_farray(self, func, order):
        def eq(got, expected):
            # Same layout, dtype, shape, etc.
            self.assertPreciseEqual(got, expected)
            # Same underlying data
            self.assertEqual(got.ctypes.data, expected.ctypes.data)

        base = np.arange(6).reshape((2, 3)).astype(np.float32).copy(order=order)

        # With typed pointer and implied dtype
        a = func(self.make_float32_pointer(base), base.shape)
        eq(a, base)
        # Integer shape
        a = func(self.make_float32_pointer(base), base.size)
        eq(a, base.ravel('K'))

        # With typed pointer and explicit dtype
        a = func(self.make_float32_pointer(base), base.shape, base.dtype)
        eq(a, base)
        a = func(self.make_float32_pointer(base), base.shape, np.float32)
        eq(a, base)

        # With voidptr and explicit dtype
        a = func(self.make_voidptr(base), base.shape, base.dtype)
        eq(a, base)
        a = func(self.make_voidptr(base), base.shape, np.int32)
        eq(a, base.view(np.int32))

        # voidptr without dtype
        with self.assertRaises(TypeError):
            func(self.make_voidptr(base), base.shape)
        # Invalid pointer type
        with self.assertRaises(TypeError):
            func(base.ctypes.data, base.shape)
        # Mismatching dtype
        with self.assertRaises(TypeError) as raises:
            func(self.make_float32_pointer(base), base.shape, np.int32)
        self.assertIn("mismatching dtype 'int32' for pointer",
                      str(raises.exception))

    @tag('important')
    def test_carray(self):
        """
        Test pure Python carray().
        """
        self.check_carray_farray(carray, 'C')

    def test_farray(self):
        """
        Test pure Python farray().
        """
        self.check_carray_farray(farray, 'F')

    def make_carray_sigs(self, formal_sig):
        """
        Generate a bunch of concrete signatures by varying the width
        and signedness of size arguments (see issue #1923).
        """
        for actual_size in (types.intp, types.int32, types.intc,
                            types.uintp, types.uint32, types.uintc):
            args = tuple(actual_size if a == types.intp else a
                         for a in formal_sig.args)
            yield formal_sig.return_type(*args)

    def check_numba_carray_farray(self, usecase, dtype_usecase):
        # With typed pointers and implicit dtype
        pyfunc = usecase
        for sig in self.make_carray_sigs(carray_float32_usecase_sig):
            f = cfunc(sig)(pyfunc)
            self.check_carray_usecase(self.make_float32_pointer, pyfunc, f.ctypes)

        # With typed pointers and explicit (matching) dtype
        pyfunc = dtype_usecase
        for sig in self.make_carray_sigs(carray_float32_usecase_sig):
            f = cfunc(sig)(pyfunc)
            self.check_carray_usecase(self.make_float32_pointer, pyfunc, f.ctypes)
        # With typed pointers and mismatching dtype
        with self.assertTypingError() as raises:
            f = cfunc(carray_float64_usecase_sig)(pyfunc)
        self.assertIn("mismatching dtype 'float32' for pointer type 'float64*'",
                      str(raises.exception))

        # With voidptr
        pyfunc = dtype_usecase
        for sig in self.make_carray_sigs(carray_voidptr_usecase_sig):
            f = cfunc(sig)(pyfunc)
            self.check_carray_usecase(self.make_float32_pointer, pyfunc, f.ctypes)

    @tag('important')
    def test_numba_carray(self):
        """
        Test Numba-compiled carray() against pure Python carray()
        """
        self.check_numba_carray_farray(carray_usecase, carray_dtype_usecase)

    def test_numba_farray(self):
        """
        Test Numba-compiled farray() against pure Python farray()
        """
        self.check_numba_carray_farray(farray_usecase, farray_dtype_usecase)


if __name__ == "__main__":
    unittest.main()
