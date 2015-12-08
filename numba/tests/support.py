"""
Assorted utilities for use in tests.
"""

import cmath
import contextlib
import errno
import math
import os
import sys
import tempfile

import numpy as np

from numba import config, errors, typing, utils, numpy_support
from numba.compiler import compile_extra, compile_isolated, Flags, DEFAULT_FLAGS
from numba.targets import cpu
import numba.unittest_support as unittest
from numba.runtime import rtsys


enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

force_pyobj_flags = Flags()
force_pyobj_flags.set("force_pyobject")

no_pyobj_flags = Flags()


is_on_numpy_16 = numpy_support.version == (1, 6)
skip_on_numpy_16 = unittest.skipIf(is_on_numpy_16,
                                   "test requires Numpy 1.7 or later")


class CompilationCache(object):
    """
    A cache of compilation results for various signatures and flags.
    This can make tests significantly faster (or less slow).
    """

    def __init__(self):
        self.typingctx = typing.Context()
        self.targetctx = cpu.CPUContext(self.typingctx)
        self.cr_cache = {}

    def compile(self, func, args, return_type=None, flags=DEFAULT_FLAGS):
        """
        Compile the function or retrieve an already compiled result
        from the cache.
        """
        cache_key = (func, args, return_type, flags)
        try:
            cr = self.cr_cache[cache_key]
        except KeyError:
            cr = compile_extra(self.typingctx, self.targetctx, func,
                               args, return_type, flags, locals={})
            self.cr_cache[cache_key] = cr
        return cr


class TestCase(unittest.TestCase):

    longMessage = True

    # A random state yielding the same random numbers for any test case.
    # Use as `self.random.<method name>`
    @utils.cached_property
    def random(self):
        return np.random.RandomState(42)

    def reset_module_warnings(self, module):
        """
        Reset the warnings registry of a module.  This can be necessary
        as the warnings module is buggy in that regard.
        See http://bugs.python.org/issue4180
        """
        if isinstance(module, str):
            module = sys.modules[module]
        try:
            del module.__warningregistry__
        except AttributeError:
            pass

    @contextlib.contextmanager
    def assertTypingError(self):
        """
        A context manager that asserts the enclosed code block fails
        compiling in nopython mode.
        """
        _accepted_errors = (errors.LoweringError, errors.TypingError,
                            TypeError, NotImplementedError)
        with self.assertRaises(_accepted_errors) as cm:
            yield cm

    @contextlib.contextmanager
    def assertRefCount(self, *objects):
        """
        A context manager that asserts the given objects have the
        same reference counts before and after executing the
        enclosed blocks.
        """
        old_refcounts = [sys.getrefcount(x) for x in objects]
        yield
        new_refcounts = [sys.getrefcount(x) for x in objects]
        for old, new, obj in zip(old_refcounts, new_refcounts, objects):
            if old != new:
                self.fail("Refcount changed from %d to %d for object: %r"
                          % (old, new, obj))

    _exact_typesets = [(bool, np.bool_), utils.INT_TYPES, (str,), (np.integer,), (utils.text_type), ]
    _approx_typesets = [(float,), (complex,), (np.inexact)]
    _sequence_typesets = [(tuple, list)]
    _float_types = (float, np.floating)
    _complex_types = (complex, np.complexfloating)

    def _detect_family(self, numeric_object):
        """
        This function returns a string description of the type family
        that the object in question belongs to.  Possible return values
        are: "exact", "complex", "approximate", "sequence", and "unknown"
        """
        if isinstance(numeric_object, np.ndarray):
            return "ndarray"

        for tp in self._sequence_typesets:
            if isinstance(numeric_object, tp):
                return "sequence"

        for tp in self._exact_typesets:
            if isinstance(numeric_object, tp):
                return "exact"

        for tp in self._complex_types:
            if isinstance(numeric_object, tp):
                return "complex"

        for tp in self._approx_typesets:
            if isinstance(numeric_object, tp):
                return "approximate"

        return "unknown"

    def _fix_dtype(self, dtype):
        """
        Fix the given *dtype* for comparison.
        """
        # Under 64-bit Windows, Numpy may return either int32 or int64
        # arrays depending on the function.
        if (sys.platform == 'win32' and sys.maxsize > 2**32 and
            dtype == np.dtype('int32')):
            return np.dtype('int64')
        else:
            return dtype

    def _fix_strides(self, arr):
        """
        Return the strides of the given array, fixed for comparison.
        Strides for 0- or 1-sized dimensions are ignored.
        """
        return [stride / arr.itemsize
                for (stride, shape) in zip(arr.strides, arr.shape)
                if shape > 1]

    def assertStridesEqual(self, first, second):
        """
        Test that two arrays have the same shape and strides.
        """
        self.assertEqual(first.shape, second.shape, "shapes differ")
        self.assertEqual(first.itemsize, second.itemsize, "itemsizes differ")
        self.assertEqual(self._fix_strides(first), self._fix_strides(second),
                         "strides differ")

    def assertPreciseEqual(self, first, second, prec='exact', ulps=1,
                           msg=None, ignore_sign_on_zero=False,
                           abs_tol=None
                           ):
        """
        Versatile equality testing function with more built-in checks than
        standard assertEqual().

        For arrays, test that layout, dtype, shape are identical, and
        recursively call assertPreciseEqual() on the contents.

        For other sequences, recursively call assertPreciseEqual() on
        the contents.

        For scalars, test that two scalars or have similar types and are
        equal up to a computed precision.
        If the scalars are instances of exact types or if *prec* is
        'exact', they are compared exactly.
        If the scalars are instances of inexact types (float, complex)
        and *prec* is not 'exact', then the number of significant bits
        is computed according to the value of *prec*: 53 bits if *prec*
        is 'double', 24 bits if *prec* is single.  This number of bits
        can be lowered by raising the *ulps* value.
        ignore_sign_on_zero can be set to True if zeros are to be considered
        equal regardless of their sign bit.
        abs_tol if this is set to a float value its value is used in the
        following. If, however, this is set to the string "eps" then machine
        precision of the type(first) is used in the following instead. This
        kwarg is used to check if the absolute difference in value between first
        and second is less than the value set, if so the numbers being compared
        are considered equal. (This is to handle small numbers typically of
        magnitude less than machine precision).

        Any value of *prec* other than 'exact', 'single' or 'double'
        will raise an error.
        """
        try:
            self._assertPreciseEqual(first, second, prec, ulps, msg,
                ignore_sign_on_zero, abs_tol)
        except AssertionError as exc:
            failure_msg = str(exc)
            # Fall off of the 'except' scope to avoid Python 3 exception
            # chaining.
        else:
            return
        # Decorate the failure message with more information
        self.fail("when comparing %s and %s: %s" % (first, second, failure_msg))

    def _assertPreciseEqual(self, first, second, prec='exact', ulps=1,
                            msg=None, ignore_sign_on_zero=False,
                            abs_tol=None):
        """Recursive workhorse for assertPreciseEqual()."""

        def _assertNumberEqual(first, second, delta=None):
            if (delta is None or first == second == 0.0
                or math.isinf(first) or math.isinf(second)):
                self.assertEqual(first, second, msg=msg)
                # For signed zeros
                if not ignore_sign_on_zero:
                    try:
                        if math.copysign(1, first) != math.copysign(1, second):
                            self.fail(
                                self._formatMessage(msg,
                                                    "%s != %s" %
                                                    (first, second)))
                    except TypeError:
                        pass
            else:
                self.assertAlmostEqual(first, second, delta=delta, msg=msg)

        first_family = self._detect_family(first)
        second_family = self._detect_family(second)

        assertion_message = "Type Family mismatch. (%s != %s)" % (first_family,
            second_family)
        if msg:
            assertion_message += ': %s' % (msg,)
        self.assertEqual(first_family, second_family, msg=assertion_message)

        # We now know they are in the same comparison family
        compare_family = first_family

        # For recognized sequences, recurse
        if compare_family == "ndarray":
            dtype = self._fix_dtype(first.dtype)
            self.assertEqual(dtype, self._fix_dtype(second.dtype))
            self.assertEqual(first.ndim, second.ndim,
                             "different number of dimensions")
            self.assertEqual(first.shape, second.shape,
                             "different shapes")
            self.assertEqual(first.flags.writeable, second.flags.writeable,
                             "different mutability")
            # itemsize is already checked by the dtype test above
            self.assertEqual(self._fix_strides(first),
                self._fix_strides(second), "different strides")
            if first.dtype != dtype:
                first = first.astype(dtype)
            if second.dtype != dtype:
                second = second.astype(dtype)
            for a, b in zip(first.flat, second.flat):
                self._assertPreciseEqual(a, b, prec, ulps, msg,
                                         ignore_sign_on_zero, abs_tol)
            return

        if compare_family == "sequence":
            self.assertEqual(len(first), len(second), msg=msg)
            for a, b in zip(first, second):
                self._assertPreciseEqual(a, b, prec, ulps, msg,
                                         ignore_sign_on_zero, abs_tol)
            return

        if compare_family == "exact":
            exact_comparison = True

        if compare_family in ["complex", "approximate"]:
            exact_comparison = False

        if compare_family == "unknown":
            # Assume these are non-numeric types: we will fall back
            # on regular unittest comparison.
            self.assertIs(first.__class__, second.__class__)
            exact_comparison = True

        # If a Numpy scalar, check the dtype is exactly the same too
        # (required for datetime64 and timedelta64).
        if hasattr(first, 'dtype') and hasattr(second, 'dtype'):
            self.assertEqual(first.dtype, second.dtype)

        try:
            if cmath.isnan(first) and cmath.isnan(second):
                # The NaNs will compare unequal, skip regular comparison
                return
        except TypeError:
            # Not floats.
            pass

        # if absolute comparison is set, use it
        if abs_tol is not None:
            if abs_tol == "eps":
                rtol = np.finfo(type(first)).eps
            elif isinstance(abs_tol, float):
                rtol = abs_tol
            else:
                raise ValueError("abs_tol is not \"eps\" or a float, found %s"
                    % abs_tol)
            if abs(first - second) < rtol:
                return

        exact_comparison = exact_comparison or prec == 'exact'

        if not exact_comparison and prec != 'exact':
            if prec == 'single':
                bits = 24
            elif prec == 'double':
                bits = 53
            else:
                raise ValueError("unsupported precision %r" % (prec,))
            k = 2 ** (ulps - bits - 1)
            delta = k * (abs(first) + abs(second))
        else:
            delta = None
        if isinstance(first, self._complex_types):
            _assertNumberEqual(first.real, second.real, delta)
            _assertNumberEqual(first.imag, second.imag, delta)
        else:
            _assertNumberEqual(first, second, delta)

    def run_nullary_func(self, pyfunc, flags):
        """
        Compile the 0-argument *pyfunc* with the given *flags*, and check
        it returns the same result as the pure Python function.
        The got and expected results are returned.
        """
        cr = compile_isolated(pyfunc, (), flags=flags)
        cfunc = cr.entry_point
        expected = pyfunc()
        got = cfunc()
        self.assertPreciseEqual(got, expected)
        return got, expected

# Various helpers

@contextlib.contextmanager
def override_config(name, value):
    """
    Return a context manager that temporarily sets Numba config variable
    *name* to *value*.  *name* must be the name of an existing variable
    in numba.config.
    """
    old_value = getattr(config, name)
    setattr(config, name, value)
    try:
        yield
    finally:
        setattr(config, name, old_value)


def compile_function(name, code, globs):
    """
    Given a *code* string, compile it with globals *globs* and return
    the function named *name*.
    """
    co = compile(code, "<string>", "exec")
    ns = {}
    eval(co, globs, ns)
    return ns[name]

def tweak_code(func, codestring=None, consts=None):
    """
    Tweak the code object of the given function by replacing its
    *codestring* (a bytes object) and *consts* tuple, optionally.
    """
    co = func.__code__
    tp = type(co)
    if codestring is None:
        codestring = co.co_code
    if consts is None:
        consts = co.co_consts
    if sys.version_info >= (3,):
        new_code = tp(co.co_argcount, co.co_kwonlyargcount, co.co_nlocals,
                      co.co_stacksize, co.co_flags, codestring,
                      consts, co.co_names, co.co_varnames,
                      co.co_filename, co.co_name, co.co_firstlineno,
                      co.co_lnotab)
    else:
        new_code = tp(co.co_argcount, co.co_nlocals,
                      co.co_stacksize, co.co_flags, codestring,
                      consts, co.co_names, co.co_varnames,
                      co.co_filename, co.co_name, co.co_firstlineno,
                      co.co_lnotab)
    func.__code__ = new_code

def static_temp_directory(dirname):
    """
    Create a directory in the temp dir with a given name. Statically-named
    temp dirs created using this function are needed because we can't delete a
    DLL under Windows (this is a bit fragile if stale files can influence the
    result of future test runs).
    """
    tmpdir = os.path.join(tempfile.gettempdir(), dirname)
    try:
        os.mkdir(tmpdir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    return tmpdir

# From CPython

@contextlib.contextmanager
def captured_output(stream_name):
    """Return a context manager used by captured_stdout/stdin/stderr
    that temporarily replaces the sys stream *stream_name* with a StringIO."""
    orig_stdout = getattr(sys, stream_name)
    setattr(sys, stream_name, utils.StringIO())
    try:
        yield getattr(sys, stream_name)
    finally:
        setattr(sys, stream_name, orig_stdout)

def captured_stdout():
    """Capture the output of sys.stdout:

       with captured_stdout() as stdout:
           print("hello")
       self.assertEqual(stdout.getvalue(), "hello\n")
    """
    return captured_output("stdout")

def captured_stderr():
    """Capture the output of sys.stderr:

       with captured_stderr() as stderr:
           print("hello", file=sys.stderr)
       self.assertEqual(stderr.getvalue(), "hello\n")
    """
    return captured_output("stderr")


class MemoryLeak(object):

    __enable_leak_check = True

    def memory_leak_setup(self):
        self.__init_stats = rtsys.get_allocation_stats()

    def memory_leak_teardown(self):
        if self.__enable_leak_check:
            self.assert_no_memory_leak()

    def assert_no_memory_leak(self):
        old = self.__init_stats
        new = rtsys.get_allocation_stats()
        total_alloc = new.alloc - old.alloc
        total_free = new.free - old.free
        total_mi_alloc = new.mi_alloc - old.mi_alloc
        total_mi_free = new.mi_free - old.mi_free
        self.assertEqual(total_alloc, total_free)
        self.assertEqual(total_mi_alloc, total_mi_free)

    def disable_leak_check(self):
        # For per-test use when MemoryLeakMixin is injected into a TestCase
        self.__enable_leak_check = False


class MemoryLeakMixin(MemoryLeak):

    def setUp(self):
        super(MemoryLeakMixin, self).setUp()
        self.memory_leak_setup()

    def tearDown(self):
        super(MemoryLeakMixin, self).tearDown()
        self.memory_leak_teardown()
