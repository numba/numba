import errno
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
import threading
import warnings
import inspect
import pickle
import weakref
from itertools import chain
from io import StringIO

import numpy as np

from numba import njit, jit, generated_jit, typeof
from numba.core import types, errors, codegen
from numba import _dispatcher
from numba.core.compiler import compile_isolated
from numba.core.errors import NumbaWarning
from numba.tests.support import (TestCase, temp_directory, import_dynamic,
                                 override_env_config, capture_cache_log,
                                 captured_stdout)
from numba.np.numpy_support import as_dtype
from numba.core.caching import _UserWideCacheLocator
from numba.core.dispatcher import Dispatcher
from numba.tests.support import (skip_parfors_unsupported, needs_lapack,
                                 SerialMixin)
from numba.testing.main import _TIMEOUT as _RUNNER_TIMEOUT
import llvmlite.binding as ll
import unittest
from numba.parfors import parfor


_TEST_TIMEOUT = _RUNNER_TIMEOUT - 60.


try:
    import jinja2
except ImportError:
    jinja2 = None

try:
    import pygments
except ImportError:
    pygments = None

_is_armv7l = platform.machine() == 'armv7l'


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


def generated_usecase(x, y=5):
    if isinstance(x, types.Complex):
        def impl(x, y):
            return x + y
    else:
        def impl(x, y):
            return x - y
    return impl


def bad_generated_usecase(x, y=5):
    if isinstance(x, types.Complex):
        def impl(x):
            return x
    else:
        def impl(x, y=6):
            return x - y
    return impl


def dtype_generated_usecase(a, b, dtype=None):
    if isinstance(dtype, (types.misc.NoneType, types.misc.Omitted)):
        out_dtype = np.result_type(*(np.dtype(ary.dtype.name)
                                   for ary in (a, b)))
    elif isinstance(dtype, (types.DType, types.NumberClass)):
        out_dtype = as_dtype(dtype)
    else:
        raise TypeError("Unhandled Type %s" % type(dtype))

    def _fn(a, b, dtype=None):
        return np.ones(a.shape, dtype=out_dtype)

    return _fn


class BaseTest(TestCase):

    jit_args = dict(nopython=True)

    def compile_func(self, pyfunc):
        def check(*args, **kwargs):
            expected = pyfunc(*args, **kwargs)
            result = f(*args, **kwargs)
            self.assertPreciseEqual(result, expected)
        f = jit(**self.jit_args)(pyfunc)
        return f, check


def check_access_is_preventable():
    # This exists to check whether it is possible to prevent access to
    # a file/directory through the use of `chmod 500`. If a user has
    # elevated rights (e.g. root) then writes are likely to be possible
    # anyway. Tests that require functioning access prevention are
    # therefore skipped based on the result of this check.
    tempdir = temp_directory('test_cache')
    test_dir = (os.path.join(tempdir, 'writable_test'))
    os.mkdir(test_dir)
    # assume access prevention is not possible
    ret = False
    # check a write is possible
    with open(os.path.join(test_dir, 'write_ok'), 'wt') as f:
        f.write('check1')
    # now forbid access
    os.chmod(test_dir, 0o500)
    try:
        with open(os.path.join(test_dir, 'write_forbidden'), 'wt') as f:
            f.write('check2')
    except (OSError, IOError) as e:
        # Check that the cause of the exception is due to access/permission
        # as per
        # https://github.com/conda/conda/blob/4.5.0/conda/gateways/disk/permissions.py#L35-L37  # noqa: E501
        eno = getattr(e, 'errno', None)
        if eno in (errno.EACCES, errno.EPERM):
            # errno reports access/perm fail so access prevention via
            # `chmod 500` works for this user.
            ret = True
    finally:
        os.chmod(test_dir, 0o775)
        shutil.rmtree(test_dir)
    return ret


_access_preventable = check_access_is_preventable()
_access_msg = "Cannot create a directory to which writes are preventable"
skip_bad_access = unittest.skipUnless(_access_preventable, _access_msg)


class TestDispatcher(BaseTest):

    def test_equality(self):
        @jit
        def foo(x):
            return x

        @jit
        def bar(x):
            return x

        # Written this way to verify `==` returns a bool (gh-5838). Using
        # `assertTrue(foo == foo)` or `assertEqual(foo, foo)` would defeat the
        # purpose of this test.
        self.assertEqual(foo == foo, True)
        self.assertEqual(foo == bar, False)
        self.assertEqual(foo == None, False)  # noqa: E711

    def test_dyn_pyfunc(self):
        @jit
        def foo(x):
            return x

        foo(1)
        [cr] = foo.overloads.values()
        # __module__ must be match that of foo
        self.assertEqual(cr.entry_point.__module__, foo.py_func.__module__)

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
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=wrapper) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertFalse(errors)

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
            r"Ambiguous overloading for <function add [^>]*> "
            r"\(float64, float64\):\n"
            r"\(float32, float64\) -> float64\n"
            r"\(float64, float32\) -> float64"
        )
        # The integer signature is not part of the best matches
        self.assertNotIn("int64", str(cm.exception))

    def test_signature_mismatch(self):
        tmpl = ("Signature mismatch: %d argument types given, but function "
                "takes 2 arguments")
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

    def test_disabled_compilation(self):
        @jit
        def foo(a):
            return a

        foo.compile("(float32,)")
        foo.disable_compile()
        with self.assertRaises(RuntimeError) as raises:
            foo.compile("(int32,)")
        self.assertEqual(str(raises.exception), "compilation disabled")
        self.assertEqual(len(foo.signatures), 1)

    def test_disabled_compilation_through_list(self):
        @jit(["(float32,)", "(int32,)"])
        def foo(a):
            return a

        with self.assertRaises(RuntimeError) as raises:
            foo.compile("(complex64,)")
        self.assertEqual(str(raises.exception), "compilation disabled")
        self.assertEqual(len(foo.signatures), 2)

    def test_disabled_compilation_nested_call(self):
        @jit(["(intp,)"])
        def foo(a):
            return a

        @jit
        def bar():
            foo(1)
            foo(np.ones(1))  # no matching definition

        with self.assertRaises(TypeError) as raises:
            bar()
        m = "No matching definition for argument type(s) array(float64, 1d, C)"
        self.assertEqual(str(raises.exception), m)

    def test_fingerprint_failure(self):
        """
        Failure in computing the fingerprint cannot affect a nopython=False
        function.  On the other hand, with nopython=True, a ValueError should
        be raised to report the failure with fingerprint.
        """
        @jit
        def foo(x):
            return x

        # Empty list will trigger failure in compile_fingerprint
        errmsg = 'cannot compute fingerprint of empty list'
        with self.assertRaises(ValueError) as raises:
            _dispatcher.compute_fingerprint([])
        self.assertIn(errmsg, str(raises.exception))
        # It should work in fallback
        self.assertEqual(foo([]), [])
        # But, not in nopython=True
        strict_foo = jit(nopython=True)(foo.py_func)
        with self.assertRaises(ValueError) as raises:
            strict_foo([])
        self.assertIn(errmsg, str(raises.exception))

        # Test in loop lifting context
        @jit
        def bar():
            object()  # force looplifting
            x = []
            for i in range(10):
                x = foo(x)
            return x

        self.assertEqual(bar(), [])
        # Make sure it was looplifted
        [cr] = bar.overloads.values()
        self.assertEqual(len(cr.lifted), 1)

    def test_serialization(self):
        """
        Test serialization of Dispatcher objects
        """
        @jit(nopython=True)
        def foo(x):
            return x + 1

        self.assertEqual(foo(1), 2)

        # get serialization memo
        memo = Dispatcher._memo
        Dispatcher._recent.clear()
        memo_size = len(memo)

        # pickle foo and check memo size
        serialized_foo = pickle.dumps(foo)
        # increases the memo size
        self.assertEqual(memo_size + 1, len(memo))

        # unpickle
        foo_rebuilt = pickle.loads(serialized_foo)
        self.assertEqual(memo_size + 1, len(memo))

        self.assertIs(foo, foo_rebuilt)

        # do we get the same object even if we delete all the explicit
        # references?
        id_orig = id(foo_rebuilt)
        del foo
        del foo_rebuilt
        self.assertEqual(memo_size + 1, len(memo))
        new_foo = pickle.loads(serialized_foo)
        self.assertEqual(id_orig, id(new_foo))

        # now clear the recent cache
        ref = weakref.ref(new_foo)
        del new_foo
        Dispatcher._recent.clear()
        self.assertEqual(memo_size, len(memo))

        # show that deserializing creates a new object
        pickle.loads(serialized_foo)
        self.assertIs(ref(), None)

    @needs_lapack
    @unittest.skipIf(_is_armv7l, "Unaligned loads unsupported")
    def test_misaligned_array_dispatch(self):
        # for context see issue #2937
        def foo(a):
            return np.linalg.matrix_power(a, 1)

        jitfoo = jit(nopython=True)(foo)

        n = 64
        r = int(np.sqrt(n))
        dt = np.int8
        count = np.complex128().itemsize // dt().itemsize

        tmp = np.arange(n * count + 1, dtype=dt)

        # create some arrays as Cartesian production of:
        # [F/C] x [aligned/misaligned]
        C_contig_aligned = tmp[:-1].view(np.complex128).reshape(r, r)
        C_contig_misaligned = tmp[1:].view(np.complex128).reshape(r, r)
        F_contig_aligned = C_contig_aligned.T
        F_contig_misaligned = C_contig_misaligned.T

        # checking routine
        def check(name, a):
            a[:, :] = np.arange(n, dtype=np.complex128).reshape(r, r)
            expected = foo(a)
            got = jitfoo(a)
            np.testing.assert_allclose(expected, got)

        # The checks must be run in this order to create the dispatch key
        # sequence that causes invalid dispatch noted in #2937.
        # The first two should hit the cache as they are aligned, supported
        # order and under 5 dimensions. The second two should end up in the
        # fallback path as they are misaligned.
        check("C_contig_aligned", C_contig_aligned)
        check("F_contig_aligned", F_contig_aligned)
        check("C_contig_misaligned", C_contig_misaligned)
        check("F_contig_misaligned", F_contig_misaligned)

    @unittest.skipIf(_is_armv7l, "Unaligned loads unsupported")
    def test_immutability_in_array_dispatch(self):

        # RO operation in function
        def foo(a):
            return np.sum(a)

        jitfoo = jit(nopython=True)(foo)

        n = 64
        r = int(np.sqrt(n))
        dt = np.int8
        count = np.complex128().itemsize // dt().itemsize

        tmp = np.arange(n * count + 1, dtype=dt)

        # create some arrays as Cartesian production of:
        # [F/C] x [aligned/misaligned]
        C_contig_aligned = tmp[:-1].view(np.complex128).reshape(r, r)
        C_contig_misaligned = tmp[1:].view(np.complex128).reshape(r, r)
        F_contig_aligned = C_contig_aligned.T
        F_contig_misaligned = C_contig_misaligned.T

        # checking routine
        def check(name, a, disable_write_bit=False):
            a[:, :] = np.arange(n, dtype=np.complex128).reshape(r, r)
            if disable_write_bit:
                a.flags.writeable = False
            expected = foo(a)
            got = jitfoo(a)
            np.testing.assert_allclose(expected, got)

        # all of these should end up in the fallback path as they have no write
        # bit set
        check("C_contig_aligned", C_contig_aligned, disable_write_bit=True)
        check("F_contig_aligned", F_contig_aligned, disable_write_bit=True)
        check("C_contig_misaligned", C_contig_misaligned,
              disable_write_bit=True)
        check("F_contig_misaligned", F_contig_misaligned,
              disable_write_bit=True)

    @needs_lapack
    @unittest.skipIf(_is_armv7l, "Unaligned loads unsupported")
    def test_misaligned_high_dimension_array_dispatch(self):

        def foo(a):
            return np.linalg.matrix_power(a[0, 0, 0, 0, :, :], 1)

        jitfoo = jit(nopython=True)(foo)

        def check_properties(arr, layout, aligned):
            self.assertEqual(arr.flags.aligned, aligned)
            if layout == "C":
                self.assertEqual(arr.flags.c_contiguous, True)
            if layout == "F":
                self.assertEqual(arr.flags.f_contiguous, True)

        n = 729
        r = 3
        dt = np.int8
        count = np.complex128().itemsize // dt().itemsize

        tmp = np.arange(n * count + 1, dtype=dt)

        # create some arrays as Cartesian production of:
        # [F/C] x [aligned/misaligned]
        C_contig_aligned = tmp[:-1].view(np.complex128).\
            reshape(r, r, r, r, r, r)
        check_properties(C_contig_aligned, 'C', True)
        C_contig_misaligned = tmp[1:].view(np.complex128).\
            reshape(r, r, r, r, r, r)
        check_properties(C_contig_misaligned, 'C', False)
        F_contig_aligned = C_contig_aligned.T
        check_properties(F_contig_aligned, 'F', True)
        F_contig_misaligned = C_contig_misaligned.T
        check_properties(F_contig_misaligned, 'F', False)

        # checking routine
        def check(name, a):
            a[:, :] = np.arange(n, dtype=np.complex128).\
                reshape(r, r, r, r, r, r)
            expected = foo(a)
            got = jitfoo(a)
            np.testing.assert_allclose(expected, got)

        # these should all hit the fallback path as the cache is only for up to
        # 5 dimensions
        check("F_contig_misaligned", F_contig_misaligned)
        check("C_contig_aligned", C_contig_aligned)
        check("F_contig_aligned", F_contig_aligned)
        check("C_contig_misaligned", C_contig_misaligned)

    def test_dispatch_recompiles_for_scalars(self):
        # for context #3612, essentially, compiling a lambda x:x for a
        # numerically wide type (everything can be converted to a complex128)
        # and then calling again with e.g. an int32 would lead to the int32
        # being converted to a complex128 whereas it ought to compile an int32
        # specialization.
        def foo(x):
            return x

        # jit and compile on dispatch for 3 scalar types, expect 3 signatures
        jitfoo = jit(nopython=True)(foo)
        jitfoo(np.complex128(1 + 2j))
        jitfoo(np.int32(10))
        jitfoo(np.bool_(False))
        self.assertEqual(len(jitfoo.signatures), 3)
        expected_sigs = [(types.complex128,), (types.int32,), (types.bool_,)]
        self.assertEqual(jitfoo.signatures, expected_sigs)

        # now jit with signatures so recompilation is forbidden
        # expect 1 signature and type conversion
        jitfoo = jit([(types.complex128,)], nopython=True)(foo)
        jitfoo(np.complex128(1 + 2j))
        jitfoo(np.int32(10))
        jitfoo(np.bool_(False))
        self.assertEqual(len(jitfoo.signatures), 1)
        expected_sigs = [(types.complex128,)]
        self.assertEqual(jitfoo.signatures, expected_sigs)

    def test_dispatcher_raises_for_invalid_decoration(self):
        # For context see https://github.com/numba/numba/issues/4750.

        @jit(nopython=True)
        def foo(x):
            return x

        with self.assertRaises(TypeError) as raises:
            jit(foo)
        err_msg = str(raises.exception)
        self.assertIn(
            "A jit decorator was called on an already jitted function", err_msg)
        self.assertIn("foo", err_msg)
        self.assertIn(".py_func", err_msg)

        with self.assertRaises(TypeError) as raises:
            jit(BaseTest)
        err_msg = str(raises.exception)
        self.assertIn("The decorated object is not a function", err_msg)
        self.assertIn(f"{type(BaseTest)}", err_msg)


class TestSignatureHandling(BaseTest):
    """
    Test support for various parameter passing styles.
    """

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


class TestSignatureHandlingObjectMode(TestSignatureHandling):
    """
    Sams as TestSignatureHandling, but in object mode.
    """

    jit_args = dict(forceobj=True)


class TestGeneratedDispatcher(TestCase):
    """
    Tests for @generated_jit.
    """

    def test_generated(self):
        f = generated_jit(nopython=True)(generated_usecase)
        self.assertEqual(f(8), 8 - 5)
        self.assertEqual(f(x=8), 8 - 5)
        self.assertEqual(f(x=8, y=4), 8 - 4)
        self.assertEqual(f(1j), 5 + 1j)
        self.assertEqual(f(1j, 42), 42 + 1j)
        self.assertEqual(f(x=1j, y=7), 7 + 1j)

    def test_generated_dtype(self):
        f = generated_jit(nopython=True)(dtype_generated_usecase)
        a = np.ones((10,), dtype=np.float32)
        b = np.ones((10,), dtype=np.float64)
        self.assertEqual(f(a, b).dtype, np.float64)
        self.assertEqual(f(a, b, dtype=np.dtype('int32')).dtype, np.int32)
        self.assertEqual(f(a, b, dtype=np.int32).dtype, np.int32)

    def test_signature_errors(self):
        """
        Check error reporting when implementation signature doesn't match
        generating function signature.
        """
        f = generated_jit(nopython=True)(bad_generated_usecase)
        # Mismatching # of arguments
        with self.assertRaises(TypeError) as raises:
            f(1j)
        self.assertIn("should be compatible with signature '(x, y=5)', "
                      "but has signature '(x)'",
                      str(raises.exception))
        # Mismatching defaults
        with self.assertRaises(TypeError) as raises:
            f(1)
        self.assertIn("should be compatible with signature '(x, y=5)', "
                      "but has signature '(x, y=6)'",
                      str(raises.exception))


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

    def _check_cfg_display(self, cfg, wrapper=''):
        # simple stringify test
        if wrapper:
            wrapper = "{}{}".format(len(wrapper), wrapper)
        module_name = __name__.split('.', 1)[0]
        module_len = len(module_name)
        prefix = r'^digraph "CFG for \'_ZN{}{}{}'.format(wrapper,
                                                         module_len,
                                                         module_name)
        self.assertRegexpMatches(str(cfg), prefix)
        # .display() requires an optional dependency on `graphviz`.
        # just test for the attribute without running it.
        self.assertTrue(callable(cfg.display))

    def test_inspect_cfg(self):
        # Exercise the .inspect_cfg(). These are minimal tests and do not fully
        # check the correctness of the function.
        @jit
        def foo(the_array):
            return the_array.sum()

        # Generate 3 overloads
        a1 = np.ones(1)
        a2 = np.ones((1, 1))
        a3 = np.ones((1, 1, 1))
        foo(a1)
        foo(a2)
        foo(a3)

        # Call inspect_cfg() without arguments
        cfgs = foo.inspect_cfg()

        # Correct count of overloads
        self.assertEqual(len(cfgs), 3)

        # Makes sure all the signatures are correct
        [s1, s2, s3] = cfgs.keys()
        self.assertEqual(set([s1, s2, s3]),
                         set(map(lambda x: (typeof(x),), [a1, a2, a3])))

        for cfg in cfgs.values():
            self._check_cfg_display(cfg)
        self.assertEqual(len(list(cfgs.values())), 3)

        # Call inspect_cfg(signature)
        cfg = foo.inspect_cfg(signature=foo.signatures[0])
        self._check_cfg_display(cfg)

    def test_inspect_cfg_with_python_wrapper(self):
        # Exercise the .inspect_cfg() including the python wrapper.
        # These are minimal tests and do not fully check the correctness of
        # the function.
        @jit
        def foo(the_array):
            return the_array.sum()

        # Generate 3 overloads
        a1 = np.ones(1)
        a2 = np.ones((1, 1))
        a3 = np.ones((1, 1, 1))
        foo(a1)
        foo(a2)
        foo(a3)

        # Call inspect_cfg(signature, show_wrapper="python")
        cfg = foo.inspect_cfg(signature=foo.signatures[0],
                              show_wrapper="python")
        self._check_cfg_display(cfg, wrapper='cpython')

    def test_inspect_types(self):
        @jit
        def foo(a, b):
            return a + b

        foo(1, 2)
        # Exercise the method
        foo.inspect_types(StringIO())

        # Test output
        expected = str(foo.overloads[foo.signatures[0]].type_annotation)
        with captured_stdout() as out:
            foo.inspect_types()
        assert expected in out.getvalue()

    def test_inspect_types_with_signature(self):
        @jit
        def foo(a):
            return a + 1

        foo(1)
        foo(1.0)
        # Inspect all signatures
        with captured_stdout() as total:
            foo.inspect_types()
        # Inspect first signature
        with captured_stdout() as first:
            foo.inspect_types(signature=foo.signatures[0])
        # Inspect second signature
        with captured_stdout() as second:
            foo.inspect_types(signature=foo.signatures[1])

        self.assertEqual(total.getvalue(), first.getvalue() + second.getvalue())

    @unittest.skipIf(jinja2 is None, "please install the 'jinja2' package")
    @unittest.skipIf(pygments is None, "please install the 'pygments' package")
    def test_inspect_types_pretty(self):
        @jit
        def foo(a, b):
            return a + b

        foo(1, 2)

        # Exercise the method, dump the output
        with captured_stdout():
            ann = foo.inspect_types(pretty=True)

        # ensure HTML <span> is found in the annotation output
        for k, v in ann.ann.items():
            span_found = False
            for line in v['pygments_lines']:
                if 'span' in line[2]:
                    span_found = True
            self.assertTrue(span_found)

        # check that file+pretty kwarg combo raises
        with self.assertRaises(ValueError) as raises:
            foo.inspect_types(file=StringIO(), pretty=True)

        self.assertIn("`file` must be None if `pretty=True`",
                      str(raises.exception))

    def test_get_annotation_info(self):
        @jit
        def foo(a):
            return a + 1

        foo(1)
        foo(1.3)

        expected = dict(chain.from_iterable(foo.get_annotation_info(i).items()
                                            for i in foo.signatures))
        result = foo.get_annotation_info()
        self.assertEqual(expected, result)

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


class BaseCacheTest(TestCase):
    # This class is also used in test_cfunc.py.

    # The source file that will be copied
    usecases_file = None
    # Make sure this doesn't conflict with another module
    modname = None

    def setUp(self):
        self.tempdir = temp_directory('test_cache')
        sys.path.insert(0, self.tempdir)
        self.modfile = os.path.join(self.tempdir, self.modname + ".py")
        self.cache_dir = os.path.join(self.tempdir, "__pycache__")
        shutil.copy(self.usecases_file, self.modfile)
        self.maxDiff = None

    def tearDown(self):
        sys.modules.pop(self.modname, None)
        sys.path.remove(self.tempdir)

    def import_module(self):
        # Import a fresh version of the test module.  All jitted functions
        # in the test module will start anew and load overloads from
        # the on-disk cache if possible.
        old = sys.modules.pop(self.modname, None)
        if old is not None:
            # Make sure cached bytecode is removed
            cached = [old.__cached__]
            for fn in cached:
                try:
                    os.unlink(fn)
                except OSError as e:
                    if e.errno != errno.ENOENT:
                        raise
        mod = import_dynamic(self.modname)
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

    def check_pycache(self, n):
        c = self.cache_contents()
        self.assertEqual(len(c), n, c)

    def dummy_test(self):
        pass


class BaseCacheUsecasesTest(BaseCacheTest):
    here = os.path.dirname(__file__)
    usecases_file = os.path.join(here, "cache_usecases.py")
    modname = "dispatcher_caching_test_fodder"

    def run_in_separate_process(self):
        # Cached functions can be run from a distinct process.
        # Also stresses issue #1603: uncached function calling cached function
        # shouldn't fail compiling.
        code = """if 1:
            import sys

            sys.path.insert(0, %(tempdir)r)
            mod = __import__(%(modname)r)
            mod.self_test()
            """ % dict(tempdir=self.tempdir, modname=self.modname)

        popen = subprocess.Popen([sys.executable, "-c", code],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = popen.communicate()
        if popen.returncode != 0:
            raise AssertionError(
                "process failed with code %s: \n"
                "stdout follows\n%s\n"
                "stderr follows\n%s\n"
                % (popen.returncode, out.decode(), err.decode()),
            )

    def check_module(self, mod):
        self.check_pycache(0)
        f = mod.add_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_pycache(2)  # 1 index, 1 data
        self.assertPreciseEqual(f(2.5, 3), 6.5)
        self.check_pycache(3)  # 1 index, 2 data

        f = mod.add_objmode_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_pycache(5)  # 2 index, 3 data
        self.assertPreciseEqual(f(2.5, 3), 6.5)
        self.check_pycache(6)  # 2 index, 4 data

        mod.self_test()

    def check_hits(self, func, hits, misses=None):
        st = func.stats
        self.assertEqual(sum(st.cache_hits.values()), hits, st.cache_hits)
        if misses is not None:
            self.assertEqual(sum(st.cache_misses.values()), misses,
                             st.cache_misses)


class TestCache(BaseCacheUsecasesTest):

    def test_caching(self):
        self.check_pycache(0)
        mod = self.import_module()
        self.check_pycache(0)

        f = mod.add_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_pycache(2)  # 1 index, 1 data
        self.assertPreciseEqual(f(2.5, 3), 6.5)
        self.check_pycache(3)  # 1 index, 2 data
        self.check_hits(f, 0, 2)

        f = mod.add_objmode_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_pycache(5)  # 2 index, 3 data
        self.assertPreciseEqual(f(2.5, 3), 6.5)
        self.check_pycache(6)  # 2 index, 4 data
        self.check_hits(f, 0, 2)

        f = mod.record_return
        rec = f(mod.aligned_arr, 1)
        self.assertPreciseEqual(tuple(rec), (2, 43.5))
        rec = f(mod.packed_arr, 1)
        self.assertPreciseEqual(tuple(rec), (2, 43.5))
        self.check_pycache(9)  # 3 index, 6 data
        self.check_hits(f, 0, 2)

        f = mod.generated_usecase
        self.assertPreciseEqual(f(3, 2), 1)
        self.assertPreciseEqual(f(3j, 2), 2 + 3j)

        # Check the code runs ok from another process
        self.run_in_separate_process()

    def test_caching_nrt_pruned(self):
        self.check_pycache(0)
        mod = self.import_module()
        self.check_pycache(0)

        f = mod.add_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_pycache(2)  # 1 index, 1 data
        # NRT pruning may affect cache
        self.assertPreciseEqual(f(2, np.arange(3)), 2 + np.arange(3) + 1)
        self.check_pycache(3)  # 1 index, 2 data
        self.check_hits(f, 0, 2)

    def test_inner_then_outer(self):
        # Caching inner then outer function is ok
        mod = self.import_module()
        self.assertPreciseEqual(mod.inner(3, 2), 6)
        self.check_pycache(2)  # 1 index, 1 data
        # Uncached outer function shouldn't fail (issue #1603)
        f = mod.outer_uncached
        self.assertPreciseEqual(f(3, 2), 2)
        self.check_pycache(2)  # 1 index, 1 data
        mod = self.import_module()
        f = mod.outer_uncached
        self.assertPreciseEqual(f(3, 2), 2)
        self.check_pycache(2)  # 1 index, 1 data
        # Cached outer will create new cache entries
        f = mod.outer
        self.assertPreciseEqual(f(3, 2), 2)
        self.check_pycache(4)  # 2 index, 2 data
        self.assertPreciseEqual(f(3.5, 2), 2.5)
        self.check_pycache(6)  # 2 index, 4 data

    def test_outer_then_inner(self):
        # Caching outer then inner function is ok
        mod = self.import_module()
        self.assertPreciseEqual(mod.outer(3, 2), 2)
        self.check_pycache(4)  # 2 index, 2 data
        self.assertPreciseEqual(mod.outer_uncached(3, 2), 2)
        self.check_pycache(4)  # same
        mod = self.import_module()
        f = mod.inner
        self.assertPreciseEqual(f(3, 2), 6)
        self.check_pycache(4)  # same
        self.assertPreciseEqual(f(3.5, 2), 6.5)
        self.check_pycache(5)  # 2 index, 3 data

    def test_no_caching(self):
        mod = self.import_module()

        f = mod.add_nocache_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_pycache(0)

    def test_looplifted(self):
        # Loop-lifted functions can't be cached and raise a warning
        mod = self.import_module()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', NumbaWarning)

            f = mod.looplifted
            self.assertPreciseEqual(f(4), 6)
            self.check_pycache(0)

        self.assertEqual(len(w), 1)
        self.assertIn('Cannot cache compiled function "looplifted" '
                      'as it uses lifted code', str(w[0].message))

    def test_big_array(self):
        # Code references big array globals cannot be cached
        mod = self.import_module()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', NumbaWarning)

            f = mod.use_big_array
            np.testing.assert_equal(f(), mod.biggie)
            self.check_pycache(0)

        self.assertEqual(len(w), 1)
        self.assertIn('Cannot cache compiled function "use_big_array" '
                      'as it uses dynamic globals', str(w[0].message))

    def test_ctypes(self):
        # Functions using a ctypes pointer can't be cached and raise
        # a warning.
        mod = self.import_module()

        for f in [mod.use_c_sin, mod.use_c_sin_nest1, mod.use_c_sin_nest2]:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always', NumbaWarning)

                self.assertPreciseEqual(f(0.0), 0.0)
                self.check_pycache(0)

            self.assertEqual(len(w), 1)
            self.assertIn(
                'Cannot cache compiled function "{}"'.format(f.__name__),
                str(w[0].message),
            )

    def test_closure(self):
        mod = self.import_module()

        with warnings.catch_warnings():
            warnings.simplefilter('error', NumbaWarning)

            f = mod.closure1
            self.assertPreciseEqual(f(3), 6) # 3 + 3 = 6
            f = mod.closure2
            self.assertPreciseEqual(f(3), 8) # 3 + 5 = 8
            f = mod.closure3
            self.assertPreciseEqual(f(3), 10) # 3 + 7 = 8
            f = mod.closure4
            self.assertPreciseEqual(f(3), 12) # 3 + 9 = 12
            self.check_pycache(5) # 1 nbi, 4 nbc

    def test_cache_reuse(self):
        mod = self.import_module()
        mod.add_usecase(2, 3)
        mod.add_usecase(2.5, 3.5)
        mod.add_objmode_usecase(2, 3)
        mod.outer_uncached(2, 3)
        mod.outer(2, 3)
        mod.record_return(mod.packed_arr, 0)
        mod.record_return(mod.aligned_arr, 1)
        mod.generated_usecase(2, 3)
        mtimes = self.get_cache_mtimes()
        # Two signatures compiled
        self.check_hits(mod.add_usecase, 0, 2)

        mod2 = self.import_module()
        self.assertIsNot(mod, mod2)
        f = mod2.add_usecase
        f(2, 3)
        self.check_hits(f, 1, 0)
        f(2.5, 3.5)
        self.check_hits(f, 2, 0)
        f = mod2.add_objmode_usecase
        f(2, 3)
        self.check_hits(f, 1, 0)

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

    def test_frozen(self):
        from .dummy_module import function
        old_code = function.__code__
        code_obj = compile('pass', 'tests/dummy_module.py', 'exec')
        try:
            function.__code__ = code_obj

            source = inspect.getfile(function)
            # doesn't return anything, since it cannot find the module
            # fails unless the executable is frozen
            locator = _UserWideCacheLocator.from_function(function, source)
            self.assertIsNone(locator)

            sys.frozen = True
            # returns a cache locator object, only works when the executable
            # is frozen
            locator = _UserWideCacheLocator.from_function(function, source)
            self.assertIsInstance(locator, _UserWideCacheLocator)

        finally:
            function.__code__ = old_code
            del sys.frozen

    def _test_pycache_fallback(self):
        """
        With a disabled __pycache__, test there is a working fallback
        (e.g. on the user-wide cache dir)
        """
        mod = self.import_module()
        f = mod.add_usecase
        # Remove this function's cache files at the end, to avoid accumulation
        # across test calls.
        self.addCleanup(shutil.rmtree, f.stats.cache_path, ignore_errors=True)

        self.assertPreciseEqual(f(2, 3), 6)
        # It's a cache miss since the file was copied to a new temp location
        self.check_hits(f, 0, 1)

        # Test re-use
        mod2 = self.import_module()
        f = mod2.add_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_hits(f, 1, 0)

        # The __pycache__ is empty (otherwise the test's preconditions
        # wouldn't be met)
        self.check_pycache(0)

    @skip_bad_access
    @unittest.skipIf(os.name == "nt",
                     "cannot easily make a directory read-only on Windows")
    def test_non_creatable_pycache(self):
        # Make it impossible to create the __pycache__ directory
        old_perms = os.stat(self.tempdir).st_mode
        os.chmod(self.tempdir, 0o500)
        self.addCleanup(os.chmod, self.tempdir, old_perms)

        self._test_pycache_fallback()

    @skip_bad_access
    @unittest.skipIf(os.name == "nt",
                     "cannot easily make a directory read-only on Windows")
    def test_non_writable_pycache(self):
        # Make it impossible to write to the __pycache__ directory
        pycache = os.path.join(self.tempdir, '__pycache__')
        os.mkdir(pycache)
        old_perms = os.stat(pycache).st_mode
        os.chmod(pycache, 0o500)
        self.addCleanup(os.chmod, pycache, old_perms)

        self._test_pycache_fallback()

    def test_ipython(self):
        # Test caching in an IPython session
        base_cmd = [sys.executable, '-m', 'IPython']
        base_cmd += ['--quiet', '--quick', '--no-banner', '--colors=NoColor']
        try:
            ver = subprocess.check_output(base_cmd + ['--version'])
        except subprocess.CalledProcessError as e:
            self.skipTest("ipython not available: return code %d"
                          % e.returncode)
        ver = ver.strip().decode()
        print("ipython version:", ver)
        # Create test input
        inputfn = os.path.join(self.tempdir, "ipython_cache_usecase.txt")
        with open(inputfn, "w") as f:
            f.write(r"""
                import os
                import sys

                from numba import jit

                # IPython 5 does not support multiline input if stdin isn't
                # a tty (https://github.com/ipython/ipython/issues/9752)
                f = jit(cache=True)(lambda: 42)

                res = f()
                # IPython writes on stdout, so use stderr instead
                sys.stderr.write(u"cache hits = %d\n" % f.stats.cache_hits[()])

                # IPython hijacks sys.exit(), bypass it
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(res)
                """)

        def execute_with_input():
            # Feed the test input as stdin, to execute it in REPL context
            with open(inputfn, "rb") as stdin:
                p = subprocess.Popen(base_cmd, stdin=stdin,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     universal_newlines=True)
                out, err = p.communicate()
                if p.returncode != 42:
                    self.fail("unexpected return code %d\n"
                              "-- stdout:\n%s\n"
                              "-- stderr:\n%s\n"
                              % (p.returncode, out, err))
                return err

        execute_with_input()
        # Run a second time and check caching
        err = execute_with_input()
        self.assertIn("cache hits = 1", err.strip())


@skip_parfors_unsupported
class TestSequentialParForsCache(BaseCacheUsecasesTest):
    def setUp(self):
        super(TestSequentialParForsCache, self).setUp()
        # Turn on sequential parfor lowering
        parfor.sequential_parfor_lowering = True

    def tearDown(self):
        super(TestSequentialParForsCache, self).tearDown()
        # Turn off sequential parfor lowering
        parfor.sequential_parfor_lowering = False

    def test_caching(self):
        mod = self.import_module()
        self.check_pycache(0)
        f = mod.parfor_usecase
        ary = np.ones(10)
        self.assertPreciseEqual(f(ary), ary * ary + ary)
        dynamic_globals = [cres.library.has_dynamic_globals
                           for cres in f.overloads.values()]
        self.assertEqual(dynamic_globals, [False])
        self.check_pycache(2)  # 1 index, 1 data


class TestCacheWithCpuSetting(BaseCacheUsecasesTest):
    # Disable parallel testing due to envvars modification
    _numba_parallel_test_ = False

    def check_later_mtimes(self, mtimes_old):
        match_count = 0
        for k, v in self.get_cache_mtimes().items():
            if k in mtimes_old:
                self.assertGreaterEqual(v, mtimes_old[k])
                match_count += 1
        self.assertGreater(match_count, 0,
                           msg='nothing to compare')

    def test_user_set_cpu_name(self):
        self.check_pycache(0)
        mod = self.import_module()
        mod.self_test()
        cache_size = len(self.cache_contents())

        mtimes = self.get_cache_mtimes()
        # Change CPU name to generic
        with override_env_config('NUMBA_CPU_NAME', 'generic'):
            self.run_in_separate_process()

        self.check_later_mtimes(mtimes)
        self.assertGreater(len(self.cache_contents()), cache_size)
        # Check cache index
        cache = mod.add_usecase._cache
        cache_file = cache._cache_file
        cache_index = cache_file._load_index()
        self.assertEqual(len(cache_index), 2)
        [key_a, key_b] = cache_index.keys()
        if key_a[1][1] == ll.get_host_cpu_name():
            key_host, key_generic = key_a, key_b
        else:
            key_host, key_generic = key_b, key_a
        self.assertEqual(key_host[1][1], ll.get_host_cpu_name())
        self.assertEqual(key_host[1][2], codegen.get_host_cpu_features())
        self.assertEqual(key_generic[1][1], 'generic')
        self.assertEqual(key_generic[1][2], '')

    def test_user_set_cpu_features(self):
        self.check_pycache(0)
        mod = self.import_module()
        mod.self_test()
        cache_size = len(self.cache_contents())

        mtimes = self.get_cache_mtimes()
        # Change CPU feature
        my_cpu_features = '-sse;-avx'

        system_features = codegen.get_host_cpu_features()

        self.assertNotEqual(system_features, my_cpu_features)
        with override_env_config('NUMBA_CPU_FEATURES', my_cpu_features):
            self.run_in_separate_process()
        self.check_later_mtimes(mtimes)
        self.assertGreater(len(self.cache_contents()), cache_size)
        # Check cache index
        cache = mod.add_usecase._cache
        cache_file = cache._cache_file
        cache_index = cache_file._load_index()
        self.assertEqual(len(cache_index), 2)
        [key_a, key_b] = cache_index.keys()

        if key_a[1][2] == system_features:
            key_host, key_generic = key_a, key_b
        else:
            key_host, key_generic = key_b, key_a

        self.assertEqual(key_host[1][1], ll.get_host_cpu_name())
        self.assertEqual(key_host[1][2], system_features)
        self.assertEqual(key_generic[1][1], ll.get_host_cpu_name())
        self.assertEqual(key_generic[1][2], my_cpu_features)


class TestMultiprocessCache(BaseCacheTest):

    # Nested multiprocessing.Pool raises AssertionError:
    # "daemonic processes are not allowed to have children"
    _numba_parallel_test_ = False

    here = os.path.dirname(__file__)
    usecases_file = os.path.join(here, "cache_usecases.py")
    modname = "dispatcher_caching_test_fodder"

    def test_multiprocessing(self):
        # Check caching works from multiple processes at once (#2028)
        mod = self.import_module()
        # Calling a pure Python caller of the JIT-compiled function is
        # necessary to reproduce the issue.
        f = mod.simple_usecase_caller
        n = 3
        try:
            ctx = multiprocessing.get_context('spawn')
        except AttributeError:
            ctx = multiprocessing
        pool = ctx.Pool(n)
        try:
            res = sum(pool.imap(f, range(n)))
        finally:
            pool.close()
        self.assertEqual(res, n * (n - 1) // 2)


class TestCacheFileCollision(unittest.TestCase):
    _numba_parallel_test_ = False

    here = os.path.dirname(__file__)
    usecases_file = os.path.join(here, "cache_usecases.py")
    modname = "caching_file_loc_fodder"
    source_text_1 = """
from numba import njit
@njit(cache=True)
def bar():
    return 123
"""
    source_text_2 = """
from numba import njit
@njit(cache=True)
def bar():
    return 321
"""

    def setUp(self):
        self.tempdir = temp_directory('test_cache_file_loc')
        sys.path.insert(0, self.tempdir)
        self.modname = 'module_name_that_is_unlikely'
        self.assertNotIn(self.modname, sys.modules)
        self.modname_bar1 = self.modname
        self.modname_bar2 = '.'.join([self.modname, 'foo'])
        foomod = os.path.join(self.tempdir, self.modname)
        os.mkdir(foomod)
        with open(os.path.join(foomod, '__init__.py'), 'w') as fout:
            print(self.source_text_1, file=fout)
        with open(os.path.join(foomod, 'foo.py'), 'w') as fout:
            print(self.source_text_2, file=fout)

    def tearDown(self):
        sys.modules.pop(self.modname_bar1, None)
        sys.modules.pop(self.modname_bar2, None)
        sys.path.remove(self.tempdir)

    def import_bar1(self):
        return import_dynamic(self.modname_bar1).bar

    def import_bar2(self):
        return import_dynamic(self.modname_bar2).bar

    def test_file_location(self):
        bar1 = self.import_bar1()
        bar2 = self.import_bar2()
        # Check that the cache file is named correctly
        idxname1 = bar1._cache._cache_file._index_name
        idxname2 = bar2._cache._cache_file._index_name
        self.assertNotEqual(idxname1, idxname2)
        self.assertTrue(idxname1.startswith("__init__.bar-3.py"))
        self.assertTrue(idxname2.startswith("foo.bar-3.py"))

    @unittest.skipUnless(hasattr(multiprocessing, 'get_context'),
                         'Test requires multiprocessing.get_context')
    def test_no_collision(self):
        bar1 = self.import_bar1()
        bar2 = self.import_bar2()
        with capture_cache_log() as buf:
            res1 = bar1()
        cachelog = buf.getvalue()
        # bar1 should save new index and data
        self.assertEqual(cachelog.count('index saved'), 1)
        self.assertEqual(cachelog.count('data saved'), 1)
        self.assertEqual(cachelog.count('index loaded'), 0)
        self.assertEqual(cachelog.count('data loaded'), 0)
        with capture_cache_log() as buf:
            res2 = bar2()
        cachelog = buf.getvalue()
        # bar2 should save new index and data
        self.assertEqual(cachelog.count('index saved'), 1)
        self.assertEqual(cachelog.count('data saved'), 1)
        self.assertEqual(cachelog.count('index loaded'), 0)
        self.assertEqual(cachelog.count('data loaded'), 0)
        self.assertNotEqual(res1, res2)

        try:
            # Make sure we can spawn new process without inheriting
            # the parent context.
            mp = multiprocessing.get_context('spawn')
        except ValueError:
            print("missing spawn context")

        q = mp.Queue()
        # Start new process that calls `cache_file_collision_tester`
        proc = mp.Process(target=cache_file_collision_tester,
                          args=(q, self.tempdir,
                                self.modname_bar1,
                                self.modname_bar2))
        proc.start()
        # Get results from the process
        log1 = q.get()
        got1 = q.get()
        log2 = q.get()
        got2 = q.get()
        proc.join()

        # The remote execution result of bar1() and bar2() should match
        # the one executed locally.
        self.assertEqual(got1, res1)
        self.assertEqual(got2, res2)

        # The remote should have loaded bar1 from cache
        self.assertEqual(log1.count('index saved'), 0)
        self.assertEqual(log1.count('data saved'), 0)
        self.assertEqual(log1.count('index loaded'), 1)
        self.assertEqual(log1.count('data loaded'), 1)

        # The remote should have loaded bar2 from cache
        self.assertEqual(log2.count('index saved'), 0)
        self.assertEqual(log2.count('data saved'), 0)
        self.assertEqual(log2.count('index loaded'), 1)
        self.assertEqual(log2.count('data loaded'), 1)


def cache_file_collision_tester(q, tempdir, modname_bar1, modname_bar2):
    sys.path.insert(0, tempdir)
    bar1 = import_dynamic(modname_bar1).bar
    bar2 = import_dynamic(modname_bar2).bar
    with capture_cache_log() as buf:
        r1 = bar1()
    q.put(buf.getvalue())
    q.put(r1)
    with capture_cache_log() as buf:
        r2 = bar2()
    q.put(buf.getvalue())
    q.put(r2)


class TestCacheMultipleFilesWithSignature(unittest.TestCase):
    # Regression test for https://github.com/numba/numba/issues/3658

    _numba_parallel_test_ = False

    source_text_file1 = """
from file2 import function2
"""
    source_text_file2 = """
from numba import njit

@njit('float64(float64)', cache=True)
def function1(x):
    return x

@njit('float64(float64)', cache=True)
def function2(x):
    return x
"""

    def setUp(self):
        self.tempdir = temp_directory('test_cache_file_loc')

        self.file1 = os.path.join(self.tempdir, 'file1.py')
        with open(self.file1, 'w') as fout:
            print(self.source_text_file1, file=fout)

        self.file2 = os.path.join(self.tempdir, 'file2.py')
        with open(self.file2, 'w') as fout:
            print(self.source_text_file2, file=fout)

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_caching_mutliple_files_with_signature(self):
        # Execute file1.py
        popen = subprocess.Popen([sys.executable, self.file1],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
        out, err = popen.communicate()
        msg = f"stdout:\n{out.decode()}\n\nstderr:\n{err.decode()}"
        self.assertEqual(popen.returncode, 0, msg=msg)

        # Execute file2.py
        popen = subprocess.Popen([sys.executable, self.file2],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
        out, err = popen.communicate()
        msg = f"stdout:\n{out.decode()}\n\nstderr:\n{err.decode()}"
        self.assertEqual(popen.returncode, 0, msg)


class TestDispatcherFunctionBoundaries(TestCase):
    def test_pass_dispatcher_as_arg(self):
        # Test that a Dispatcher object can be pass as argument
        @jit(nopython=True)
        def add1(x):
            return x + 1

        @jit(nopython=True)
        def bar(fn, x):
            return fn(x)

        @jit(nopython=True)
        def foo(x):
            return bar(add1, x)

        # Check dispatcher as argument inside NPM
        inputs = [1, 11.1, np.arange(10)]
        expected_results = [x + 1 for x in inputs]

        for arg, expect in zip(inputs, expected_results):
            self.assertPreciseEqual(foo(arg), expect)

        # Check dispatcher as argument from python
        for arg, expect in zip(inputs, expected_results):
            self.assertPreciseEqual(bar(add1, arg), expect)

    def test_dispatcher_as_arg_usecase(self):
        @jit(nopython=True)
        def maximum(seq, cmpfn):
            tmp = seq[0]
            for each in seq[1:]:
                cmpval = cmpfn(tmp, each)
                if cmpval < 0:
                    tmp = each
            return tmp

        got = maximum([1, 2, 3, 4], cmpfn=jit(lambda x, y: x - y))
        self.assertEqual(got, 4)
        got = maximum(list(zip(range(5), range(5)[::-1])),
                      cmpfn=jit(lambda x, y: x[0] - y[0]))
        self.assertEqual(got, (4, 0))
        got = maximum(list(zip(range(5), range(5)[::-1])),
                      cmpfn=jit(lambda x, y: x[1] - y[1]))
        self.assertEqual(got, (0, 4))

    def test_dispatcher_can_return_to_python(self):
        @jit(nopython=True)
        def foo(fn):
            return fn

        fn = jit(lambda x: x)

        self.assertEqual(foo(fn), fn)

    def test_dispatcher_in_sequence_arg(self):
        @jit(nopython=True)
        def one(x):
            return x + 1

        @jit(nopython=True)
        def two(x):
            return one(one(x))

        @jit(nopython=True)
        def three(x):
            return one(one(one(x)))

        @jit(nopython=True)
        def choose(fns, x):
            return fns[0](x), fns[1](x), fns[2](x)

        # Tuple case
        self.assertEqual(choose((one, two, three), 1), (2, 3, 4))
        # List case
        self.assertEqual(choose([one, one, one], 1), (2, 2, 2))


class TestBoxingDefaultError(unittest.TestCase):
    # Testing default error at boxing/unboxing
    def test_unbox_runtime_error(self):
        # Dummy type has no unbox support
        def foo(x):
            pass
        cres = compile_isolated(foo, (types.Dummy("dummy_type"),))
        with self.assertRaises(TypeError) as raises:
            # Can pass in whatever and the unbox logic will always raise
            # without checking the input value.
            cres.entry_point(None)
        self.assertEqual(str(raises.exception), "can't unbox dummy_type type")

    def test_box_runtime_error(self):
        def foo():
            return unittest  # Module type has no boxing logic
        cres = compile_isolated(foo, ())
        with self.assertRaises(TypeError) as raises:
            # Can pass in whatever and the unbox logic will always raise
            # without checking the input value.
            cres.entry_point()
        pat = "cannot convert native Module.* to Python object"
        self.assertRegexpMatches(str(raises.exception), pat)


class TestNoRetryFailedSignature(unittest.TestCase):
    """Test that failed-to-compile signatures are not recompiled.
    """

    def run_test(self, func):
        fcom = func._compiler
        self.assertEqual(len(fcom._failed_cache), 0)
        # expected failure because `int` has no `__getitem__`
        with self.assertRaises(errors.TypingError):
            func(1)
        self.assertEqual(len(fcom._failed_cache), 1)
        # retry
        with self.assertRaises(errors.TypingError):
            func(1)
        self.assertEqual(len(fcom._failed_cache), 1)
        # retry with double
        with self.assertRaises(errors.TypingError):
            func(1.0)
        self.assertEqual(len(fcom._failed_cache), 2)

    def test_direct_call(self):
        @jit(nopython=True)
        def foo(x):
            return x[0]

        self.run_test(foo)

    def test_nested_call(self):
        @jit(nopython=True)
        def bar(x):
            return x[0]

        @jit(nopython=True)
        def foobar(x):
            bar(x)

        @jit(nopython=True)
        def foo(x):
            return bar(x) + foobar(x)

        self.run_test(foo)

    def test_error_count(self):
        def check(field, would_fail):
            # Slightly modified from the reproducer in issue #4117.
            # Before the patch, the compilation time of the failing case is
            # much longer than of the successful case. This can be detected
            # by the number of times `trigger()` is visited.
            k = 10
            counter = {'c': 0}

            @generated_jit
            def trigger(x):
                # Keep track of every visit
                counter['c'] += 1
                if would_fail:
                    raise errors.TypingError("invoke_failed")
                return lambda x: x

            @jit(nopython=True)
            def ident(out, x):
                pass

            def chain_assign(fs, inner=ident):
                tab_head, tab_tail = fs[-1], fs[:-1]

                @jit(nopython=True)
                def assign(out, x):
                    inner(out, x)
                    out[0] += tab_head(x)

                if tab_tail:
                    return chain_assign(tab_tail, assign)
                else:
                    return assign

            chain = chain_assign((trigger,) * k)
            out = np.ones(2)
            if would_fail:
                with self.assertRaises(errors.TypingError) as raises:
                    chain(out, 1)
                self.assertIn('invoke_failed', str(raises.exception))
            else:
                chain(out, 1)

            # Returns the visit counts
            return counter['c']

        ct_ok = check('a', False)
        ct_bad = check('c', True)
        # `trigger()` is visited exactly once for both successful and failed
        # compilation.
        self.assertEqual(ct_ok, 1)
        self.assertEqual(ct_bad, 1)


@njit
def add_y1(x, y=1):
    return x + y


@njit
def add_ynone(x, y=None):
    return x + (1 if y else 2)


@njit
def mult(x, y):
    return x * y


@njit
def add_func(x, func=mult):
    return x + func(x, x)


def _checker(f1, arg):
    assert f1(arg) == f1.py_func(arg)


class TestMultiprocessingDefaultParameters(SerialMixin, unittest.TestCase):
    def run_fc_multiproc(self, fc):
        try:
            ctx = multiprocessing.get_context('spawn')
        except AttributeError:
            ctx = multiprocessing

        # RE: issue #5973, this doesn't use multiprocessing.Pool.map as doing so
        # causes the TBB library to segfault under certain conditions. It's not
        # clear whether the cause is something in the complexity of the Pool
        # itself, e.g. watcher threads etc, or if it's a problem synonymous with
        # a "timing attack".
        for a in [1, 2, 3]:
            p = ctx.Process(target=_checker, args=(fc, a,))
            p.start()
            p.join(_TEST_TIMEOUT)
            self.assertEqual(p.exitcode, 0)

    def test_int_def_param(self):
        """ Tests issue #4888"""

        self.run_fc_multiproc(add_y1)

    def test_none_def_param(self):
        """ Tests None as a default parameter"""

        self.run_fc_multiproc(add_func)

    def test_function_def_param(self):
        """ Tests a function as a default parameter"""

        self.run_fc_multiproc(add_func)


if __name__ == '__main__':
    unittest.main()
