from __future__ import print_function, division, absolute_import

import threading

import numpy as np

from numba import unittest_support as unittest
from numba import utils, vectorize, jit
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


if __name__ == '__main__':
    unittest.main()
