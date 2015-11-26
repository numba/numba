from __future__ import print_function, division, absolute_import

import errno
import imp
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import warnings

import numpy as np

from numba import unittest_support as unittest
from numba import utils, vectorize, jit
from numba.config import NumbaWarning
from .support import TestCase


def dummy(x):
    return x


def add(x, y):
    return x + y


def addsub(x, y, z):
    return x - y + z


def addsub_defaults(x, y=2, z=3):
    return x - y + z


def star_defaults(x, y=2, *z):
    return x, y, z


class TestDispatcher(TestCase):

    def compile_func(self, pyfunc):
        def check(*args, **kwargs):
            expected = pyfunc(*args, **kwargs)
            result = f(*args, **kwargs)
            self.assertPreciseEqual(result, expected)
        f = jit(nopython=True)(pyfunc)
        return f, check

    def test_numba_interface(self):
        """
        Check that vectorize can accept a decorated object.
        """
        vectorize('f8(f8)')(jit(dummy))

    def test_no_argument(self):
        @jit
        def foo():
            return 1

        # Just make sure this doesn't crash
        foo()

    def test_coerce_input_types(self):
        # Issue #486: do not allow unsafe conversions if we can still
        # compile other specializations.
        c_add = jit(nopython=True)(add)
        self.assertPreciseEqual(c_add(123, 456), add(123, 456))
        self.assertPreciseEqual(c_add(12.3, 45.6), add(12.3, 45.6))
        self.assertPreciseEqual(c_add(12.3, 45.6j), add(12.3, 45.6j))
        self.assertPreciseEqual(c_add(12300000000, 456), add(12300000000, 456))

        # Now force compilation of only a single specialization
        c_add = jit('(i4, i4)', nopython=True)(add)
        self.assertPreciseEqual(c_add(123, 456), add(123, 456))
        # Implicit (unsafe) conversion of float to int
        self.assertPreciseEqual(c_add(12.3, 45.6), add(12, 45))
        with self.assertRaises(TypeError):
            # Implicit conversion of complex to int disallowed
            c_add(12.3, 45.6j)

    def test_ambiguous_new_version(self):
        """Test compiling new version in an ambiguous case
        """
        @jit
        def foo(a, b):
            return a + b

        INT = 1
        FLT = 1.5
        self.assertAlmostEqual(foo(INT, FLT), INT + FLT)
        self.assertEqual(len(foo.overloads), 1)
        self.assertAlmostEqual(foo(FLT, INT), FLT + INT)
        self.assertEqual(len(foo.overloads), 2)
        self.assertAlmostEqual(foo(FLT, FLT), FLT + FLT)
        self.assertEqual(len(foo.overloads), 3)
        # The following call is ambiguous because (int, int) can resolve
        # to (float, int) or (int, float) with equal weight.
        self.assertAlmostEqual(foo(1, 1), INT + INT)
        self.assertEqual(len(foo.overloads), 4, "didn't compile a new "
                                                "version")

    def test_lock(self):
        """
        Test that (lazy) compiling from several threads at once doesn't
        produce errors (see issue #908).
        """
        errors = []

        @jit
        def foo(x):
            return x + 1

        def wrapper():
            try:
                self.assertEqual(foo(1), 2)
            except BaseException as e:
                errors.append(e)

        threads = [threading.Thread(target=wrapper) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertFalse(errors)

    def test_named_args(self):
        """
        Test passing named arguments to a dispatcher.
        """
        f, check = self.compile_func(addsub)
        check(3, z=10, y=4)
        check(3, 4, 10)
        check(x=3, y=4, z=10)
        # All calls above fall under the same specialization
        self.assertEqual(len(f.overloads), 1)
        # Errors
        with self.assertRaises(TypeError) as cm:
            f(3, 4, y=6, z=7)
        self.assertIn("too many arguments: expected 3, got 4",
                      str(cm.exception))
        with self.assertRaises(TypeError) as cm:
            f()
        self.assertIn("not enough arguments: expected 3, got 0",
                      str(cm.exception))
        with self.assertRaises(TypeError) as cm:
            f(3, 4, y=6)
        self.assertIn("missing argument 'z'", str(cm.exception))

    def test_default_args(self):
        """
        Test omitting arguments with a default value.
        """
        f, check = self.compile_func(addsub_defaults)
        check(3, z=10, y=4)
        check(3, 4, 10)
        check(x=3, y=4, z=10)
        # Now omitting some values
        check(3, z=10)
        check(3, 4)
        check(x=3, y=4)
        check(3)
        check(x=3)
        # All calls above fall under the same specialization
        self.assertEqual(len(f.overloads), 1)
        # Errors
        with self.assertRaises(TypeError) as cm:
            f(3, 4, y=6, z=7)
        self.assertIn("too many arguments: expected 3, got 4",
                      str(cm.exception))
        with self.assertRaises(TypeError) as cm:
            f()
        self.assertIn("not enough arguments: expected at least 1, got 0",
                      str(cm.exception))
        with self.assertRaises(TypeError) as cm:
            f(y=6, z=7)
        self.assertIn("missing argument 'x'", str(cm.exception))

    def test_star_args(self):
        """
        Test a compiled function with starargs in the signature.
        """
        f, check = self.compile_func(star_defaults)
        check(4)
        check(4, 5)
        check(4, 5, 6)
        check(4, 5, 6, 7)
        check(4, 5, 6, 7, 8)
        check(x=4)
        check(x=4, y=5)
        check(4, y=5)
        with self.assertRaises(TypeError) as cm:
            f(4, 5, y=6)
        self.assertIn("some keyword arguments unexpected", str(cm.exception))
        with self.assertRaises(TypeError) as cm:
            f(4, 5, z=6)
        self.assertIn("some keyword arguments unexpected", str(cm.exception))
        with self.assertRaises(TypeError) as cm:
            f(4, x=6)
        self.assertIn("some keyword arguments unexpected", str(cm.exception))

    def test_explicit_signatures(self):
        f = jit("(int64,int64)")(add)
        # Approximate match (unsafe conversion)
        self.assertPreciseEqual(f(1.5, 2.5), 3)
        self.assertEqual(len(f.overloads), 1, f.overloads)
        f = jit(["(int64,int64)", "(float64,float64)"])(add)
        # Exact signature matches
        self.assertPreciseEqual(f(1, 2), 3)
        self.assertPreciseEqual(f(1.5, 2.5), 4.0)
        # Approximate match (int32 -> float64 is a safe conversion)
        self.assertPreciseEqual(f(np.int32(1), 2.5), 3.5)
        # No conversion
        with self.assertRaises(TypeError) as cm:
            f(1j, 1j)
        self.assertIn("No matching definition", str(cm.exception))
        self.assertEqual(len(f.overloads), 2, f.overloads)
        # A more interesting one...
        f = jit(["(float32,float32)", "(float64,float64)"])(add)
        self.assertPreciseEqual(f(np.float32(1), np.float32(2**-25)), 1.0)
        self.assertPreciseEqual(f(1, 2**-25), 1.0000000298023224)
        # Fail to resolve ambiguity between the two best overloads
        f = jit(["(float32,float64)",
                 "(float64,float32)",
                 "(int64,int64)"])(add)
        with self.assertRaises(TypeError) as cm:
            f(1.0, 2.0)
        # The two best matches are output in the error message, as well
        # as the actual argument types.
        self.assertRegexpMatches(
            str(cm.exception),
            r"Ambiguous overloading for <function add [^>]*> \(float64, float64\):\n"
            r"\(float32, float64\) -> float64\n"
            r"\(float64, float32\) -> float64"
            )
        # The integer signature is not part of the best matches
        self.assertNotIn("int64", str(cm.exception))

    def test_signature_mismatch(self):
        tmpl = "Signature mismatch: %d argument types given, but function takes 2 arguments"
        with self.assertRaises(TypeError) as cm:
            jit("()")(add)
        self.assertIn(tmpl % 0, str(cm.exception))
        with self.assertRaises(TypeError) as cm:
            jit("(intc,)")(add)
        self.assertIn(tmpl % 1, str(cm.exception))
        with self.assertRaises(TypeError) as cm:
            jit("(intc,intc,intc)")(add)
        self.assertIn(tmpl % 3, str(cm.exception))
        # With forceobj=True, an empty tuple is accepted
        jit("()", forceobj=True)(add)
        with self.assertRaises(TypeError) as cm:
            jit("(intc,)", forceobj=True)(add)
        self.assertIn(tmpl % 1, str(cm.exception))

    def test_matching_error_message(self):
        f = jit("(intc,intc)")(add)
        with self.assertRaises(TypeError) as cm:
            f(1j, 1j)
        self.assertEqual(str(cm.exception),
                         "No matching definition for argument type(s) "
                         "complex128, complex128")


class TestDispatcherMethods(TestCase):

    def test_recompile(self):
        closure = 1

        @jit
        def foo(x):
            return x + closure
        self.assertPreciseEqual(foo(1), 2)
        self.assertPreciseEqual(foo(1.5), 2.5)
        self.assertEqual(len(foo.signatures), 2)
        closure = 2
        self.assertPreciseEqual(foo(1), 2)
        # Recompiling takes the new closure into account.
        foo.recompile()
        # Everything was recompiled
        self.assertEqual(len(foo.signatures), 2)
        self.assertPreciseEqual(foo(1), 3)
        self.assertPreciseEqual(foo(1.5), 3.5)

    def test_recompile_signatures(self):
        # Same as above, but with an explicit signature on @jit.
        closure = 1

        @jit("int32(int32)")
        def foo(x):
            return x + closure
        self.assertPreciseEqual(foo(1), 2)
        self.assertPreciseEqual(foo(1.5), 2)
        closure = 2
        self.assertPreciseEqual(foo(1), 2)
        # Recompiling takes the new closure into account.
        foo.recompile()
        self.assertPreciseEqual(foo(1), 3)
        self.assertPreciseEqual(foo(1.5), 3)

    def test_inspect_llvm(self):
        # Create a jited function
        @jit
        def foo(explicit_arg1, explicit_arg2):
            return explicit_arg1 + explicit_arg2

        # Call it in a way to create 3 signatures
        foo(1, 1)
        foo(1.0, 1)
        foo(1.0, 1.0)

        # base call to get all llvm in a dict
        llvms = foo.inspect_llvm()
        self.assertEqual(len(llvms), 3)

        # make sure the function name shows up in the llvm
        for llvm_bc in llvms.values():
            # Look for the function name
            self.assertIn("foo", llvm_bc)

            # Look for the argument names
            self.assertIn("explicit_arg1", llvm_bc)
            self.assertIn("explicit_arg2", llvm_bc)

    def test_inspect_asm(self):
        # Create a jited function
        @jit
        def foo(explicit_arg1, explicit_arg2):
            return explicit_arg1 + explicit_arg2

        # Call it in a way to create 3 signatures
        foo(1, 1)
        foo(1.0, 1)
        foo(1.0, 1.0)

        # base call to get all llvm in a dict
        asms = foo.inspect_asm()
        self.assertEqual(len(asms), 3)

        # make sure the function name shows up in the llvm
        for asm in asms.values():
            # Look for the function name
            self.assertTrue("foo" in asm)

    def test_inspect_types(self):
        @jit
        def foo(a, b):
            return a + b

        foo(1, 2)
        # Exercise the method
        foo.inspect_types(utils.StringIO())

    def test_issue_with_array_layout_conflict(self):
        """
        This test an issue with the dispatcher when an array that is both
        C and F contiguous is supplied as the first signature.
        The dispatcher checks for F contiguous first but the compiler checks
        for C contiguous first. This results in an C contiguous code inserted
        as F contiguous function.
        """
        def pyfunc(A, i, j):
            return A[i, j]

        cfunc = jit(pyfunc)

        ary_c_and_f = np.array([[1.]])
        ary_c = np.array([[0., 1.], [2., 3.]], order='C')
        ary_f = np.array([[0., 1.], [2., 3.]], order='F')

        exp_c = pyfunc(ary_c, 1, 0)
        exp_f = pyfunc(ary_f, 1, 0)

        self.assertEqual(1., cfunc(ary_c_and_f, 0, 0))
        got_c = cfunc(ary_c, 1, 0)
        got_f = cfunc(ary_f, 1, 0)

        self.assertEqual(exp_c, got_c)
        self.assertEqual(exp_f, got_f)


class TestCache(TestCase):

    here = os.path.dirname(__file__)
    # The source file that will be copied
    usecases_file = os.path.join(here, "cache_usecases.py")
    # Make sure this doesn't conflict with another module
    modname = "caching_test_fodder"

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        sys.path.insert(0, self.tempdir)
        self.modfile = os.path.join(self.tempdir, self.modname + ".py")
        self.cache_dir = os.path.join(self.tempdir, "__pycache__")
        shutil.copy(self.usecases_file, self.modfile)
        self.maxDiff = None

    def tearDown(self):
        sys.modules.pop(self.modname, None)
        sys.path.remove(self.tempdir)
        shutil.rmtree(self.tempdir)

    def import_module(self):
        # Import a fresh version of the test module
        old = sys.modules.pop(self.modname, None)
        if old is not None:
            # Make sure cached bytecode is removed
            if sys.version_info >= (3,):
                cached = [old.__cached__]
            else:
                if old.__file__.endswith(('.pyc', '.pyo')):
                    cached = [old.__file__]
                else:
                    cached = [old.__file__ + 'c', old.__file__ + 'o']
            for fn in cached:
                try:
                    os.unlink(fn)
                except OSError as e:
                    if e.errno != errno.ENOENT:
                        raise
        mod = __import__(self.modname)
        self.assertEqual(mod.__file__.rstrip('co'), self.modfile)
        return mod

    def cache_contents(self):
        try:
            return [fn for fn in os.listdir(self.cache_dir)
                    if not fn.endswith(('.pyc', ".pyo"))]
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise
            return []

    def get_cache_mtimes(self):
        return dict((fn, os.path.getmtime(os.path.join(self.cache_dir, fn)))
                    for fn in sorted(self.cache_contents()))

    def check_cache(self, n):
        c = self.cache_contents()
        self.assertEqual(len(c), n, c)

    def dummy_test(self):
        pass

    def run_in_separate_process(self):
        # Cached functions can be run from a distinct process
        code = """if 1:
            import sys

            sys.path.insert(0, %(tempdir)r)
            mod = __import__(%(modname)r)
            assert mod.add_usecase(2, 3) == 6
            assert mod.add_objmode_usecase(2, 3) == 6
            assert mod.outer(3, 2) == 2
            packed_rec = mod.record_return(mod.packed_arr, 1)
            assert tuple(packed_rec) == (2, 43.5), packed_rec
            aligned_rec = mod.record_return(mod.aligned_arr, 1)
            assert tuple(aligned_rec) == (2, 43.5), aligned_rec
            """ % dict(tempdir=self.tempdir, modname=self.modname,
                       test_class=self.__class__.__name__)

        popen = subprocess.Popen([sys.executable, "-c", code],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = popen.communicate()
        if popen.returncode != 0:
            raise AssertionError("process failed with code %s: stderr follows\n%s\n"
                                 % (popen.returncode, err.decode()))

    def check_module(self, mod):
        self.check_cache(0)
        f = mod.add_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_cache(2)  # 1 index, 1 data
        self.assertPreciseEqual(f(2.5, 3), 6.5)
        self.check_cache(3)  # 1 index, 2 data

        f = mod.add_objmode_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_cache(5)  # 2 index, 3 data
        self.assertPreciseEqual(f(2.5, 3), 6.5)
        self.check_cache(6)  # 2 index, 4 data

    def test_caching(self):
        self.check_cache(0)
        mod = self.import_module()
        self.check_cache(0)

        f = mod.add_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_cache(2)  # 1 index, 1 data
        self.assertPreciseEqual(f(2.5, 3), 6.5)
        self.check_cache(3)  # 1 index, 2 data

        f = mod.add_objmode_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_cache(5)  # 2 index, 3 data
        self.assertPreciseEqual(f(2.5, 3), 6.5)
        self.check_cache(6)  # 2 index, 4 data

        f = mod.record_return
        rec = f(mod.aligned_arr, 1)
        self.assertPreciseEqual(tuple(rec), (2, 43.5))
        rec = f(mod.packed_arr, 1)
        self.assertPreciseEqual(tuple(rec), (2, 43.5))
        self.check_cache(9)  # 3 index, 6 data

        # Check the code runs ok from another process
        self.run_in_separate_process()

    def test_inner_then_outer(self):
        # Caching inner then outer function is ok
        mod = self.import_module()
        self.assertPreciseEqual(mod.inner(3, 2), 6)
        self.check_cache(2)  # 1 index, 1 data
        f = mod.outer
        self.assertPreciseEqual(f(3, 2), 2)
        self.check_cache(4)  # 2 index, 2 data
        self.assertPreciseEqual(f(3.5, 2), 2.5)
        self.check_cache(6)  # 2 index, 4 data

    def test_outer_then_inner(self):
        # Caching outer then inner function is ok
        mod = self.import_module()
        self.assertPreciseEqual(mod.outer(3, 2), 2)
        self.check_cache(4)  # 2 index, 2 data
        f = mod.inner
        self.assertPreciseEqual(f(3, 2), 6)
        self.assertPreciseEqual(f(3.5, 2), 6.5)
        self.check_cache(5)  # 2 index, 3 data

    def test_no_caching(self):
        mod = self.import_module()

        f = mod.add_nocache_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_cache(0)

    def test_looplifted(self):
        # Loop-lifted functions can't be cached and raise a warning
        mod = self.import_module()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', NumbaWarning)

            f = mod.looplifted
            self.assertPreciseEqual(f(4), 6)
            self.check_cache(0)

        self.assertEqual(len(w), 1)
        self.assertEqual(str(w[0].message),
                         'Cannot cache compiled function "looplifted" '
                         'as it uses lifted loops')

    def test_ctypes(self):
        # Functions using a ctypes pointer can't be cached and raise
        # a warning.
        mod = self.import_module()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', NumbaWarning)

            f = mod.use_c_sin
            self.assertPreciseEqual(f(0.0), 0.0)
            self.check_cache(0)

        self.assertEqual(len(w), 1)
        self.assertIn('Cannot cache compiled function "use_c_sin"',
                      str(w[0].message))

    def test_closure(self):
        mod = self.import_module()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', NumbaWarning)

            f = mod.closure1
            self.assertPreciseEqual(f(3), 6)
            f = mod.closure2
            self.assertPreciseEqual(f(3), 8)
            self.check_cache(0)

        self.assertEqual(len(w), 2)
        for item in w:
            self.assertIn('Cannot cache compiled function "closure"',
                          str(item.message))

    def test_cache_reuse(self):
        mod = self.import_module()
        mod.add_usecase(2, 3)
        mod.add_objmode_usecase(2, 3)
        mod.outer(2, 3)
        mod.record_return(mod.packed_arr, 0)
        mod.record_return(mod.aligned_arr, 1)
        mtimes = self.get_cache_mtimes()

        mod2 = self.import_module()
        self.assertIsNot(mod, mod2)
        mod.add_usecase(2, 3)
        mod.add_objmode_usecase(2, 3)

        # The files haven't changed
        self.assertEqual(self.get_cache_mtimes(), mtimes)

        self.run_in_separate_process()
        self.assertEqual(self.get_cache_mtimes(), mtimes)

    def test_cache_invalidate(self):
        mod = self.import_module()
        f = mod.add_usecase
        self.assertPreciseEqual(f(2, 3), 6)

        # This should change the functions' results
        with open(self.modfile, "a") as f:
            f.write("\nZ = 10\n")

        mod = self.import_module()
        f = mod.add_usecase
        self.assertPreciseEqual(f(2, 3), 15)
        f = mod.add_objmode_usecase
        self.assertPreciseEqual(f(2, 3), 15)

    def test_recompile(self):
        # Explicit call to recompile() should overwrite the cache
        mod = self.import_module()
        f = mod.add_usecase
        self.assertPreciseEqual(f(2, 3), 6)

        mod = self.import_module()
        f = mod.add_usecase
        mod.Z = 10
        self.assertPreciseEqual(f(2, 3), 6)
        f.recompile()
        self.assertPreciseEqual(f(2, 3), 15)

        # Freshly recompiled version is re-used from other imports
        mod = self.import_module()
        f = mod.add_usecase
        self.assertPreciseEqual(f(2, 3), 15)

    def test_same_names(self):
        # Function with the same names should still disambiguate
        mod = self.import_module()
        f = mod.renamed_function1
        self.assertPreciseEqual(f(2), 4)
        f = mod.renamed_function2
        self.assertPreciseEqual(f(2), 8)


if __name__ == '__main__':
    unittest.main()
