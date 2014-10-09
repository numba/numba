"""
Assorted utilities for use in tests.
"""

import cmath
import contextlib
import math
import sys

import numpy as np

from numba import types, typing, utils
from numba.compiler import compile_extra, compile_isolated, Flags, DEFAULT_FLAGS
from numba.lowering import LoweringError
from numba.targets import cpu
from numba.typeinfer import TypingError
import numba.unittest_support as unittest


enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

force_pyobj_flags = Flags()
force_pyobj_flags.set("force_pyobject")

no_pyobj_flags = Flags()


skip_on_numpy_16 = unittest.skipIf(np.__version__.startswith("1.6."),
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
        with self.assertRaises(
            (LoweringError, TypingError, TypeError, NotImplementedError)) as cm:
            yield cm

    _exact_typesets = [(bool, np.bool_), utils.INT_TYPES, (str,), (utils.text_type),]
    _approx_typesets = [(float,), (complex,), (np.inexact)]
    _sequence_typesets = [(tuple,)]
    _float_types = (float, np.floating)
    _complex_types = (complex, np.complexfloating)

    def assertPreciseEqual(self, first, second, prec='exact', ulps=1,
                           msg=None):
        """
        Test that two scalars have similar types and are equal up to
        a computed precision.
        If the scalars are instances of exact types or if *prec* is
        'exact', they are compared exactly.
        If the scalars are instances of inexact types (float, complex)
        and *prec* is not 'exact', then the number of significant bits
        is computed according to the value of *prec*: 53 bits if *prec*
        is 'double', 24 bits if *prec* is single.  This number of bits
        can be lowered by raising the *ulps* value.

        Any value of *prec* other than 'exact', 'single' or 'double'
        will raise an error.
        """
        try:
            self._assertPreciseEqual(first, second, prec, ulps, msg)
        except AssertionError as exc:
            failure_msg = str(exc)
            # Fall off of the 'except' scope to avoid Python 3 exception
            # chaining.
            self.fail("when comparing %s and %s: %s" % (first, second, failure_msg))
        else:
            return
        # Decorate the failure message with more information
        self.fail("when comparing %s and %s: %s" % (first, second, failure_msg))

    def _assertPreciseEqual(self, first, second, prec='exact', ulps=1,
                            msg=None):
        """Recursive workhorse for assertPreciseEqual()."""

        def _assertNumberEqual(first, second, delta=None):
            if (delta is None or first == second == 0.0
                or math.isinf(first) or math.isinf(second)):
                self.assertEqual(first, second, msg=msg)
                # For signed zeros
                try:
                    if math.copysign(1, first) != math.copysign(1, second):
                        self.fail(
                            self._formatMessage(msg,
                                                "%s != %s" % (first, second)))
                except TypeError:
                    pass
            else:
                self.assertAlmostEqual(first, second, delta=delta, msg=msg)

        for tp in self._sequence_typesets:
            # For recognized sequences, recurse
            if isinstance(first, tp) or isinstance(second, tp):
                self.assertIsInstance(first, tp)
                self.assertIsInstance(second, tp)
                self.assertEqual(len(first), len(second), msg=msg)
                for a, b in zip(first, second):
                    self._assertPreciseEqual(a, b, prec, ulps, msg)
                return
        for tp in self._exact_typesets:
            # One or another could be the expected, the other the actual;
            # test both.
            if isinstance(first, tp) or isinstance(second, tp):
                self.assertIsInstance(first, tp)
                self.assertIsInstance(second, tp)
                exact_comparison = True
                break
        else:
            for tp in self._approx_typesets:
                if isinstance(first, tp) or isinstance(second, tp):
                    self.assertIsInstance(first, tp)
                    self.assertIsInstance(second, tp)
                    exact_comparison = False
                    break
            else:
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

