from __future__ import print_function, absolute_import

from numba.tests.support import override_config, TestCase
from numba.ocl.testing import skip_on_oclsim
from numba import unittest_support as unittest
from numba import ocl, types


class TestOclDebugInfo(TestCase):
    """
    These tests only checks the compiled PTX for debuginfo section
    """
    def _getasm(self, fn, sig):
        fn.compile(sig)
        return fn.inspect_asm(sig)

    def _check(self, fn, sig, expect):
        asm = self._getasm(fn, sig=sig)
        assertfn = self.assertIn if expect else self.assertNotIn
        assertfn('.section .debug_info {', asm, msg=asm)

    def test_no_debuginfo_in_asm(self):
        @ocl.jit(debug=False)
        def foo(x):
            x[0] = 1

        self._check(foo, sig=(types.int32[:],), expect=False)

    def test_debuginfo_in_asm(self):
        @ocl.jit(debug=True)
        def foo(x):
            x[0] = 1

        self._check(foo, sig=(types.int32[:],), expect=True)

    def test_environment_override(self):
        with override_config('OCL_DEBUGINFO_DEFAULT', 1):
            # Using default value
            @ocl.jit
            def foo(x):
                x[0] = 1

            self._check(foo, sig=(types.int32[:],), expect=True)

            # User override default value
            @ocl.jit(debug=False)
            def bar(x):
                x[0] = 1

            self._check(bar, sig=(types.int32[:],), expect=False)


if __name__ == '__main__':
    unittest.main()
