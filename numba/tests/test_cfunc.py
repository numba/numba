"""
Tests for @cfunc and friends.
"""

from __future__ import division, print_function, absolute_import

import ctypes
import os
import subprocess
import sys

from numba import unittest_support as unittest
from numba import cfunc, types, typing, utils
from numba.types.abstract import _typecache
from numba import jit, numpy_support
from .support import TestCase, tag, captured_stderr
from .test_dispatcher import BaseCacheTest


def add_usecase(a, b):
    return a + b

def div_usecase(a, b):
    return a / b

add_sig = "float64(float64, float64)"

div_sig = "float64(int64, int64)"

def objmode_usecase(a, b):
    object()
    return a + b


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
        self.assertIn("Exception ignored", err)
        self.assertIn("ZeroDivisionError: division by zero", err)

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


class TestCache(BaseCacheTest):

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


if __name__ == "__main__":
    unittest.main()
