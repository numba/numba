from math import sqrt
from numba import cuda, float32, int16, int32, uint32, void
from numba.cuda import compile_ptx, compile_ptx_for_current_device

from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase


@skip_on_cudasim('Compilation unsupported in the simulator')
class TestCompileToPTX(unittest.TestCase):
    def test_global_kernel(self):
        def f(r, x, y):
            i = cuda.grid(1)
            if i < len(r):
                r[i] = x[i] + y[i]

        args = (float32[:], float32[:], float32[:])
        ptx, resty = compile_ptx(f, args)

        # Kernels should not have a func_retval parameter
        self.assertNotIn('func_retval', ptx)
        # .visible .func is used to denote a device function
        self.assertNotIn('.visible .func', ptx)
        # .visible .entry would denote the presence of a global function
        self.assertIn('.visible .entry', ptx)
        # Return type for kernels should always be void
        self.assertEqual(resty, void)

    def test_device_function(self):
        def add(x, y):
            return x + y

        args = (float32, float32)
        ptx, resty = compile_ptx(add, args, device=True)

        # Device functions take a func_retval parameter for storing the
        # returned value in by reference
        self.assertIn('func_retval', ptx)
        # .visible .func is used to denote a device function
        self.assertIn('.visible .func', ptx)
        # .visible .entry would denote the presence of a global function
        self.assertNotIn('.visible .entry', ptx)
        # Inferred return type as expected?
        self.assertEqual(resty, float32)

        # Check that function's output matches signature
        sig_int32 = int32(int32, int32)
        ptx, resty = compile_ptx(add, sig_int32, device=True)
        self.assertEqual(resty, int32)

        sig_int16 = int16(int16, int16)
        ptx, resty = compile_ptx(add, sig_int16, device=True)
        self.assertEqual(resty, int16)
        # Using string as signature
        sig_string = "uint32(uint32, uint32)"
        ptx, resty = compile_ptx(add, sig_string, device=True)
        self.assertEqual(resty, uint32)

    def test_fastmath(self):
        def f(x, y, z, d):
            return sqrt((x * y + z) / d)

        args = (float32, float32, float32, float32)
        ptx, resty = compile_ptx(f, args, device=True)

        # Without fastmath, fma contraction is enabled by default, but ftz and
        # approximate div / sqrt is not.
        self.assertIn('fma.rn.f32', ptx)
        self.assertIn('div.rn.f32', ptx)
        self.assertIn('sqrt.rn.f32', ptx)

        ptx, resty = compile_ptx(f, args, device=True, fastmath=True)

        # With fastmath, ftz and approximate div / sqrt are enabled
        self.assertIn('fma.rn.ftz.f32', ptx)
        self.assertIn('div.approx.ftz.f32', ptx)
        self.assertIn('sqrt.approx.ftz.f32', ptx)

    def check_debug_info(self, ptx):
        # A debug_info section should exist in the PTX. Whitespace varies
        # between CUDA toolkit versions.
        self.assertRegex(ptx, '\\.section\\s+\\.debug_info')
        # A .file directive should be produced and include the name of the
        # source. The path and whitespace may vary, so we accept anything
        # ending in the filename of this module.
        self.assertRegex(ptx, '\\.file.*test_compiler.py"')

    def test_device_function_with_debug(self):
        # See Issue #6719 - this ensures that compilation with debug succeeds
        # with CUDA 11.2 / NVVM 7.0 onwards. Previously it failed because NVVM
        # IR version metadata was not added when compiling device functions,
        # and NVVM assumed DBG version 1.0 if not specified, which is
        # incompatible with the 3.0 IR we use. This was specified only for
        # kernels.
        def f():
            pass

        ptx, resty = compile_ptx(f, (), device=True, debug=True)
        self.check_debug_info(ptx)

    def test_kernel_with_debug(self):
        # Inspired by (but not originally affected by) Issue #6719
        def f():
            pass

        ptx, resty = compile_ptx(f, (), debug=True)
        self.check_debug_info(ptx)

    def check_line_info(self, ptx):
        # A .file directive should be produced and include the name of the
        # source. The path and whitespace may vary, so we accept anything
        # ending in the filename of this module.
        self.assertRegex(ptx, '\\.file.*test_compiler.py"')

    def test_device_function_with_line_info(self):
        def f():
            pass

        ptx, resty = compile_ptx(f, (), device=True, lineinfo=True)
        self.check_line_info(ptx)

    def test_kernel_with_line_info(self):
        def f():
            pass

        ptx, resty = compile_ptx(f, (), lineinfo=True)
        self.check_line_info(ptx)

    def test_non_void_return_type(self):
        def f(x, y):
            return x[0] + y[0]

        with self.assertRaisesRegex(TypeError, 'must have void return type'):
            compile_ptx(f, (uint32[::1], uint32[::1]))


@skip_on_cudasim('Compilation unsupported in the simulator')
class TestCompileToPTXForCurrentDevice(CUDATestCase):
    def test_compile_ptx_for_current_device(self):
        def add(x, y):
            return x + y

        args = (float32, float32)
        ptx, resty = compile_ptx_for_current_device(add, args, device=True)

        # Check we target the current device's compute capability, or the
        # closest compute capability supported by the current toolkit.
        device_cc = cuda.get_current_device().compute_capability
        cc = cuda.cudadrv.nvvm.find_closest_arch(device_cc)
        target = f'.target sm_{cc[0]}{cc[1]}'
        self.assertIn(target, ptx)


@skip_on_cudasim('Compilation unsupported in the simulator')
class TestCompileOnlyTests(unittest.TestCase):
    '''For tests where we can only check correctness by examining the compiler
    output rather than observing the effects of execution.'''

    def test_nanosleep(self):
        def use_nanosleep(x):
            # Sleep for a constant time
            cuda.nanosleep(32)
            # Sleep for a variable time
            cuda.nanosleep(x)

        ptx, resty = compile_ptx(use_nanosleep, (uint32,), cc=(7, 0))

        nanosleep_count = 0
        for line in ptx.split('\n'):
            if 'nanosleep.u32' in line:
                nanosleep_count += 1

        expected = 2
        self.assertEqual(expected, nanosleep_count,
                         (f'Got {nanosleep_count} nanosleep instructions, '
                          f'expected {expected}'))


if __name__ == '__main__':
    unittest.main()
