from __future__ import print_function, absolute_import

from numba.tests.support import override_config, TestCase
from numba.cuda.testing import skip_on_cudasim
from numba import unittest_support as unittest
from numba import cuda, types


@skip_on_cudasim('Simulator does not produce debug dumps')
class TestCudaDebugInfo(TestCase):
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
        @cuda.jit(debug=False)
        def foo(x):
            x[0] = 1

        self._check(foo, sig=(types.int32[:],), expect=False)

    def test_debuginfo_in_asm(self):
        @cuda.jit(debug=True)
        def foo(x):
            x[0] = 1

        self._check(foo, sig=(types.int32[:],), expect=True)

    def test_environment_override(self):
        with override_config('CUDA_DEBUGINFO_DEFAULT', 1):
            # Using default value
            @cuda.jit
            def foo(x):
                x[0] = 1

            self._check(foo, sig=(types.int32[:],), expect=True)

            # User override default value
            @cuda.jit(debug=False)
            def bar(x):
                x[0] = 1

            self._check(bar, sig=(types.int32[:],), expect=False)


if __name__ == '__main__':
    unittest.main()
