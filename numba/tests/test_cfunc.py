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
from numba.types.abstract import _typecache
from numba import jit, numpy_support
from .support import TestCase, tag, captured_stderr
from .test_dispatcher import BaseCacheTest


def add_usecase(a, b):
    return a + b

def div_usecase(a, b):
    c = a / b
    return c

add_sig = "float64(float64, float64)"

div_sig = "float64(int64, int64)"

def objmode_usecase(a, b):
    object()
    return a + b

# Test functions for carray() and farray()

def add_pointers_c(in_ptr, out_ptr, m, n):
    in_ = carray(in_ptr, (m, n))
    out = carray(out_ptr, (m, n))
    assert in_.flags.c_contiguous
    assert out.flags.c_contiguous
    for i in range(m):
        for j in range(n):
            out[i, j] = i - j + in_[i, j]

def add_pointers_f(in_ptr, out_ptr, m, n):
    in_ = farray(in_ptr, (m, n))
    out = farray(out_ptr, (m, n))
    assert in_.flags.f_contiguous
    assert out.flags.f_contiguous
    for i in range(m):
        for j in range(n):
            out[i, j] = i - j + in_[i, j]

add_pointers_sig = types.void(types.CPointer(types.float32),
                              types.CPointer(types.float32),
                              types.intp, types.intp)

def add_voidptr_c(in_ptr, out_ptr, m, n):
    in_ = carray(in_ptr, (m, n), dtype=np.float32)
    out = carray(out_ptr, (m, n), dtype=np.float32)
    assert in_.flags.c_contiguous
    assert out.flags.c_contiguous
    for i in range(m):
        for j in range(n):
            out[i, j] = i - j + in_[i, j]

def add_voidptr_f(in_ptr, out_ptr, m, n):
    in_ = farray(in_ptr, (m, n), dtype=np.float32)
    out = farray(out_ptr, (m, n), dtype=np.float32)
    assert in_.flags.f_contiguous
    assert out.flags.f_contiguous
    for i in range(m):
        for j in range(n):
            out[i, j] = i - j + in_[i, j]

add_voidptr_sig = types.void(types.voidptr, types.voidptr,
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
        if sys.version_info >= (3,):
            self.assertIn("Exception ignored", err)
            self.assertIn("ZeroDivisionError: division by zero", err)
        else:
            self.assertIn("ZeroDivisionError('division by zero',)", err)
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

    def run_add_pointers(self, pointer_factory, func):
        a = np.linspace(0.5, 2.0, 6).reshape((2, 3)).astype(np.float32)
        out = np.empty_like(a)
        func(pointer_factory(a), pointer_factory(out), *a.shape)
        return out

    def make_voidptr(self, arr):
        return arr.ctypes.data_as(ctypes.c_void_p)

    def make_typed_pointer(self, arr):
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def check_add_pointers(self, pointer_factory, pyfunc, cfunc):
        expected = self.run_add_pointers(pointer_factory, pyfunc)
        got = self.run_add_pointers(pointer_factory, cfunc)
        self.assertPreciseEqual(expected, got)

    def check_carray_farray(self, func, order):
        def eq(got, expected):
            # Same layout, dtype, shape, etc.
            self.assertPreciseEqual(got, expected)
            # Same underlying data
            self.assertEqual(got.ctypes.data, expected.ctypes.data)

        base = np.arange(6).reshape((2, 3)).astype(np.float32).copy(order=order)
        # With typed pointer and implied dtype
        a = func(self.make_typed_pointer(base), base.shape)
        eq(a, base)
        # With typed pointer and explicit dtype
        a = func(self.make_typed_pointer(base), base.shape, base.dtype)
        eq(a, base)
        a = func(self.make_typed_pointer(base), base.shape, np.int32)
        eq(a, base.view(np.int32))
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

    def test_farray(self):
        """
        Pure Python farray().
        """
        self.check_carray_farray(farray, 'F')

    def test_carray(self):
        """
        Pure Python carray().
        """
        self.check_carray_farray(carray, 'C')

    def test_numba_carray(self):
        """
        Test Numba-compiled carray() against pure Python carray()
        """
        # With typed pointers
        pyfunc = add_pointers_c
        f = cfunc(add_pointers_sig)(pyfunc)
        self.check_add_pointers(self.make_typed_pointer, pyfunc, f.ctypes)


if __name__ == "__main__":
    unittest.main()
