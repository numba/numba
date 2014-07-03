"""
Assorted utilities for use in tests.
"""

import contextlib

from numba import types, typing, utils
from numba.compiler import compile_extra, compile_isolated, Flags, DEFAULT_FLAGS
from numba.lowering import LoweringError
from numba.targets import cpu
from numba.typeinfer import TypingError
import numba.unittest_support as unittest


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

    @contextlib.contextmanager
    def assertTypingError(self):
        """
        A context manager that asserts the enclosed code block fails
        compiling in nopython mode.
        """
        with self.assertRaises(
            (LoweringError, TypingError, TypeError, NotImplementedError)) as cm:
            yield cm

    _exact_typesets = [(bool,), utils.INT_TYPES, (str,), (utils.unicode),]
    _approx_typesets = [(float,), (complex,)]

    def assertPreciseEqual(self, first, second, prec='exact', msg=None):
        """
        Test that two scalars have similar types and are equal up to
        a computed precision.
        If the scalars are instances of exact types or if *prec* is
        'exact', they are compared exactly.
        If the scalars are instances of inexact types (float, complex)
        and *prec* is not 'exact', then the number of significant bits
        is computed according to the value of *prec*: 53 bits if *prec*
        is 'double', 24 bits if *prec* is single.

        Any value of *prec* other than 'exact', 'single' or 'double'
        will raise an error.
        """
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
                self.fail("unsupported types: %r and %r"
                          % (type(first), type(second)))

        if not exact_comparison and prec != 'exact':
            if prec == 'single':
                k = 2**-24
            elif prec == 'double':
                k = 2**-53
            else:
                raise ValueError("unsupported precision %r" % (prec,))
            delta = k * (abs(first) + abs(second))
            self.assertAlmostEqual(first, second, delta=delta, msg=msg)
        else:
            self.assertEqual(first, second, msg=msg)
