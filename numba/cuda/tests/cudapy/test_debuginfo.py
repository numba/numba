from numba.tests.support import override_config
from numba.cuda.testing import skip_on_cudasim
from numba import cuda
from numba.cuda.cudadrv.nvvm import NVVM
from numba.core import types
from numba.cuda.testing import CUDATestCase
import re
import unittest


@skip_on_cudasim('Simulator does not produce debug dumps')
class TestCudaDebugInfo(CUDATestCase):
    """
    These tests only checks the compiled PTX for debuginfo section
    """
    def _getasm(self, fn, sig):
        fn.compile(sig)
        return fn.inspect_asm(sig)

    def _check(self, fn, sig, expect):
        asm = self._getasm(fn, sig=sig)
        re_section_dbginfo = re.compile(r"\.section\s+\.debug_info\s+{")
        match = re_section_dbginfo.search(asm)
        assertfn = self.assertIsNotNone if expect else self.assertIsNone
        assertfn(match, msg=asm)

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

    def test_issue_5835(self):
        # Invalid debug metadata would segfault NVVM when any function was
        # compiled with debug turned on and optimization off. This eager
        # compilation should not crash anything.
        @cuda.jit((types.int32[::1],), debug=True, opt=False)
        def f(x):
            x[0] = 0

    def test_wrapper_has_debuginfo(self):
        sig = (types.int32[::1],)

        @cuda.jit(sig, debug=True, opt=0)
        def f(x):
            x[0] = 1

        llvm_ir = f.inspect_llvm(sig)

        if NVVM().is_nvvm70:
            # NNVM 7.0 IR attaches a debug metadata reference to the
            # definition
            defines = [line for line in llvm_ir.splitlines()
                       if 'define void @"_ZN6cudapy' in line]

            # Make sure we only found one definition
            self.assertEqual(len(defines), 1)

            wrapper_define = defines[0]
            self.assertIn('noinline!dbg', wrapper_define)
        else:
            # NVVM 3.4 subprogram debuginfo refers to the definition.
            # '786478' is a constant referring to a subprogram.
            disubprograms = [line for line in llvm_ir.splitlines()
                             if '786478' in line]

            # Make sure we only found one subprogram
            self.assertEqual(len(disubprograms), 1)

            wrapper_disubprogram = disubprograms[0]
            # Check that the subprogram points to a wrapper (these are all in
            # the "cudapy::" namespace).
            self.assertIn('_ZN6cudapy', wrapper_disubprogram)

    def test_debug_function_calls_internal_impl(self):
        # Calling a function in a module generated from an implementation
        # internal to Numba reqires multiple modules to be compiled with NVVM -
        # the internal implementation, and the caller. This example uses two
        # modules because the `in (2, 3)` is implemented with:
        #
        # numba::cpython::listobj::in_seq::$3clocals$3e::seq_contains_impl$242(
        #     UniTuple<long long, 2>,
        #     int
        # )
        #
        # This is condensed from the reproducer from c200chromebook in Issue
        # #5311.

        @cuda.jit((types.int32[:], types.int32[:]), debug=True)
        def f(inp, outp):
            outp[0] = 1 if inp[0] in (2, 3) else 3

    def test_debug_function_calls_device_function(self):
        # Calling a device function requires compilation of multiple modules
        # with NVVM - one for the caller and one for the callee. This checks
        # that we don't cause an NVVM error in this case.

        @cuda.jit(device=True, debug=True, opt=0)
        def threadid():
            return cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

        @cuda.jit((types.int32[:],), debug=True, opt=0)
        def kernel(arr):
            i = cuda.grid(1)
            if i < len(arr):
                arr[i] = threadid()


if __name__ == '__main__':
    unittest.main()
