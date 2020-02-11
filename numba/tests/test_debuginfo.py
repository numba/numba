import re

from numba.tests.support import TestCase, override_config
from numba import jit
from numba.core import types
import unittest


class TestDebugInfo(TestCase):
    """
    These tests only checks the compiled assembly for debuginfo.
    """
    def _getasm(self, fn, sig):
        fn.compile(sig)
        return fn.inspect_asm(sig)

    def _check(self, fn, sig, expect):
        asm = self._getasm(fn, sig=sig)
        m = re.search(r"\.section.+debug", asm, re.I)
        got = m is not None
        self.assertEqual(expect, got, msg='debug info not found in:\n%s' % asm)

    def test_no_debuginfo_in_asm(self):
        @jit(nopython=True, debug=False)
        def foo(x):
            return x

        self._check(foo, sig=(types.int32,), expect=False)

    def test_debuginfo_in_asm(self):
        @jit(nopython=True, debug=True)
        def foo(x):
            return x

        self._check(foo, sig=(types.int32,), expect=True)

    def test_environment_override(self):
        with override_config('DEBUGINFO_DEFAULT', 1):
            # Using default value
            @jit(nopython=True)
            def foo(x):
                return x
            self._check(foo, sig=(types.int32,), expect=True)

            # User override default
            @jit(nopython=True, debug=False)
            def bar(x):
                return x
            self._check(bar, sig=(types.int32,), expect=False)


if __name__ == '__main__':
    unittest.main()
