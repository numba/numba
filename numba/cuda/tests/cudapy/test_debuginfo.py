from numba.tests.support import override_config
from numba.cuda.testing import skip_on_cudasim
from numba import cuda
from numba.cuda.cudadrv.nvvm import NVVM
from numba.core import types
from numba.core.errors import NumbaInvalidConfigWarning
from numba.cuda.testing import CUDATestCase
import itertools
import re
import unittest
import warnings


@skip_on_cudasim('Simulator does not produce debug dumps')
class TestCudaDebugInfo(CUDATestCase):
    """
    These tests only checks the compiled PTX for debuginfo section
    """
    def _getasm(self, fn, sig):
        fn.compile(sig)
        return fn.inspect_asm(sig)

    def _check(self, fn, sig, expect):
        if not NVVM().is_nvvm70:
            self.skipTest("debuginfo not generated for NVVM 3.4")

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
        @cuda.jit(debug=True, opt=False)
        def foo(x):
            x[0] = 1

        self._check(foo, sig=(types.int32[:],), expect=True)

    def test_environment_override(self):
        with override_config('CUDA_DEBUGINFO_DEFAULT', 1):
            # Using default value
            @cuda.jit(opt=False)
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
            self.assertIn('noinline !dbg', wrapper_define)
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
        # internal to Numba requires multiple modules to be compiled with NVVM -
        # the internal implementation, and the caller. This example uses two
        # modules because the `in (2, 3)` is implemented with:
        #
        # numba::cpython::listobj::in_seq::$3clocals$3e::seq_contains_impl$242(
        #     UniTuple<long long, 2>,
        #     int
        # )
        #
        # This is condensed from this reproducer in Issue 5311:
        # https://github.com/numba/numba/issues/5311#issuecomment-674206587

        @cuda.jit((types.int32[:], types.int32[:]), debug=True, opt=False)
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

    def _test_chained_device_function(self, kernel_debug, f1_debug, f2_debug):
        @cuda.jit(device=True, debug=f2_debug, opt=False)
        def f2(x):
            return x + 1

        @cuda.jit(device=True, debug=f1_debug, opt=False)
        def f1(x, y):
            return x - f2(y)

        @cuda.jit((types.int32, types.int32), debug=kernel_debug, opt=False)
        def kernel(x, y):
            f1(x, y)

        kernel[1, 1](1, 2)

    def test_chained_device_function(self):
        # Calling a device function that calls another device function from a
        # kernel with should succeed regardless of which jit decorators have
        # debug=True. See Issue #7159.

        debug_opts = itertools.product(*[(True, False)] * 3)

        for kernel_debug, f1_debug, f2_debug in debug_opts:
            with self.subTest(kernel_debug=kernel_debug,
                              f1_debug=f1_debug,
                              f2_debug=f2_debug):
                self._test_chained_device_function(kernel_debug,
                                                   f1_debug,
                                                   f2_debug)

    def _test_chained_device_function_two_calls(self, kernel_debug, f1_debug,
                                                f2_debug):

        @cuda.jit(device=True, debug=f2_debug, opt=False)
        def f2(x):
            return x + 1

        @cuda.jit(device=True, debug=f1_debug, opt=False)
        def f1(x, y):
            return x - f2(y)

        @cuda.jit(debug=kernel_debug, opt=False)
        def kernel(x, y):
            f1(x, y)
            f2(x)

        kernel[1, 1](1, 2)

    def test_chained_device_function_two_calls(self):
        # Calling a device function that calls a leaf device function from a
        # kernel, and calling the leaf device function from the kernel should
        # succeed, regardless of which jit decorators have debug=True. See
        # Issue #7159.

        debug_opts = itertools.product(*[(True, False)] * 3)

        for kernel_debug, f1_debug, f2_debug in debug_opts:
            with self.subTest(kernel_debug=kernel_debug,
                              f1_debug=f1_debug,
                              f2_debug=f2_debug):
                self._test_chained_device_function_two_calls(kernel_debug,
                                                             f1_debug,
                                                             f2_debug)

    def check_warnings(self, warnings, warn_count):
        if NVVM().is_nvvm70:
            # We should not warn on NVVM 7.0.
            self.assertEqual(len(warnings), 0)
        else:
            self.assertEqual(len(warnings), warn_count)
            # Each warning should warn about not generating debug info.
            for warning in warnings:
                self.assertIs(warning.category, NumbaInvalidConfigWarning)
                self.assertIn('debuginfo is not generated',
                              str(warning.message))

    def test_debug_warning(self):
        # We don't generate debug info for NVVM 3.4, and warn accordingly. Here
        # we check that no warnings appear with NVVM 7.0, and that warnings
        # appear as appropriate with NVVM 3.4
        debug_opts = itertools.product(*[(True, False)] * 3)

        for kernel_debug, f1_debug, f2_debug in debug_opts:
            with self.subTest(kernel_debug=kernel_debug,
                              f1_debug=f1_debug,
                              f2_debug=f2_debug):
                with warnings.catch_warnings(record=True) as w:
                    self._test_chained_device_function_two_calls(kernel_debug,
                                                                 f1_debug,
                                                                 f2_debug)
                warn_count = kernel_debug + f1_debug + f2_debug
                self.check_warnings(w, warn_count)

    def test_chained_device_three_functions(self):
        # Like test_chained_device_function, but with enough functions (three)
        # to ensure that the recursion visits all the way down the call tree
        # when fixing linkage of functions for debug.
        def three_device_fns(kernel_debug, leaf_debug):
            @cuda.jit(device=True, debug=leaf_debug, opt=False)
            def f3(x):
                return x * x

            @cuda.jit(device=True)
            def f2(x):
                return f3(x) + 1

            @cuda.jit(device=True)
            def f1(x, y):
                return x - f2(y)

            @cuda.jit(debug=kernel_debug, opt=False)
            def kernel(x, y):
                f1(x, y)

            kernel[1, 1](1, 2)

        # Check when debug on the kernel, on the leaf, and not on any function.

        with warnings.catch_warnings(record=True) as w:
            three_device_fns(kernel_debug=True, leaf_debug=True)
        self.check_warnings(w, 2)

        with warnings.catch_warnings(record=True) as w:
            three_device_fns(kernel_debug=True, leaf_debug=False)
        self.check_warnings(w, 1)

        with warnings.catch_warnings(record=True) as w:
            three_device_fns(kernel_debug=False, leaf_debug=True)
        self.check_warnings(w, 1)

        with warnings.catch_warnings(record=True) as w:
            three_device_fns(kernel_debug=False, leaf_debug=False)
        self.check_warnings(w, 0)


if __name__ == '__main__':
    unittest.main()
