import os.path
import numpy as np
import warnings
from numba.cuda.testing import unittest
from numba.cuda.testing import (skip_on_cudasim, skip_unless_cuda_python,
                                skip_if_cuda_includes_missing,
                                skip_with_cuda_python)
from numba.cuda.testing import CUDATestCase
from numba.cuda.cudadrv.driver import Linker, LinkerError, NvrtcError
from numba.cuda import require_context
from numba.tests.support import ignore_internal_warnings
from numba import cuda, void, float64, int64


def func_with_lots_of_registers(x, a, b, c, d, e, f):
    a1 = 1.0
    a2 = 1.0
    a3 = 1.0
    a4 = 1.0
    a5 = 1.0
    b1 = 1.0
    b2 = 1.0
    b3 = 1.0
    b4 = 1.0
    b5 = 1.0
    c1 = 1.0
    c2 = 1.0
    c3 = 1.0
    c4 = 1.0
    c5 = 1.0
    d1 = 10
    d2 = 10
    d3 = 10
    d4 = 10
    d5 = 10
    for i in range(a):
        a1 += b
        a2 += c
        a3 += d
        a4 += e
        a5 += f
        b1 *= b
        b2 *= c
        b3 *= d
        b4 *= e
        b5 *= f
        c1 /= b
        c2 /= c
        c3 /= d
        c4 /= e
        c5 /= f
        d1 <<= b
        d2 <<= c
        d3 <<= d
        d4 <<= e
        d5 <<= f
    x[cuda.grid(1)] = a1 + a2 + a3 + a4 + a5
    x[cuda.grid(1)] += b1 + b2 + b3 + b4 + b5
    x[cuda.grid(1)] += c1 + c2 + c3 + c4 + c5
    x[cuda.grid(1)] += d1 + d2 + d3 + d4 + d5


@skip_on_cudasim('Linking unsupported in the simulator')
class TestLinker(CUDATestCase):

    @require_context
    def test_linker_basic(self):
        '''Simply go through the constructor and destructor
        '''
        linker = Linker.new()
        del linker

    def _test_linking(self, eager):
        global bar  # must be a global; other it is recognized as a freevar
        bar = cuda.declare_device('bar', 'int32(int32)')

        link = os.path.join(os.path.dirname(__file__), 'data', 'jitlink.ptx')

        if eager:
            args = ['void(int32[:], int32[:])']
        else:
            args = []

        @cuda.jit(*args, link=[link])
        def foo(x, y):
            i = cuda.grid(1)
            x[i] += bar(y[i])

        A = np.array([123], dtype=np.int32)
        B = np.array([321], dtype=np.int32)

        foo[1, 1](A, B)

        self.assertTrue(A[0] == 123 + 2 * 321)

    def test_linking_lazy_compile(self):
        self._test_linking(eager=False)

    def test_linking_eager_compile(self):
        self._test_linking(eager=True)

    @skip_unless_cuda_python('NVIDIA Binding needed for NVRTC')
    def test_linking_cu(self):
        bar = cuda.declare_device('bar', 'int32(int32)')

        link = os.path.join(os.path.dirname(__file__), 'data', 'jitlink.cu')

        @cuda.jit(link=[link])
        def kernel(r, x):
            i = cuda.grid(1)

            if i < len(r):
                r[i] = bar(x[i])

        x = np.arange(10, dtype=np.int32)
        r = np.zeros_like(x)

        kernel[1, 32](r, x)

        # Matches the operation of bar() in jitlink.cu
        expected = x * 2
        np.testing.assert_array_equal(r, expected)

    @skip_unless_cuda_python('NVIDIA Binding needed for NVRTC')
    def test_linking_cu_log_warning(self):
        bar = cuda.declare_device('bar', 'int32(int32)')

        link = os.path.join(os.path.dirname(__file__), 'data', 'warn.cu')

        with warnings.catch_warnings(record=True) as w:
            ignore_internal_warnings()

            @cuda.jit('void(int32)', link=[link])
            def kernel(x):
                bar(x)

        self.assertEqual(len(w), 1, 'Expected warnings from NVRTC')
        # Check the warning refers to the log messages
        self.assertIn('NVRTC log messages', str(w[0].message))
        # Check the message pertaining to the unused variable is provided
        self.assertIn('declared but never referenced', str(w[0].message))

    @skip_unless_cuda_python('NVIDIA Binding needed for NVRTC')
    def test_linking_cu_error(self):
        bar = cuda.declare_device('bar', 'int32(int32)')

        link = os.path.join(os.path.dirname(__file__), 'data', 'error.cu')

        with self.assertRaises(NvrtcError) as e:
            @cuda.jit('void(int32)', link=[link])
            def kernel(x):
                bar(x)

        msg = e.exception.args[0]
        # Check the error message refers to the NVRTC compile
        self.assertIn('NVRTC Compilation failure', msg)
        # Check the expected error in the CUDA source is reported
        self.assertIn('identifier "SYNTAX" is undefined', msg)
        # Check the filename is reported correctly
        self.assertIn('in the compilation of "error.cu"', msg)

    @skip_with_cuda_python
    def test_linking_cu_ctypes_unsupported(self):
        msg = ('Linking CUDA source files is not supported with the ctypes '
               'binding')
        with self.assertRaisesRegex(NotImplementedError, msg):
            @cuda.jit('void()', link=['jitlink.cu'])
            def f():
                pass

    def test_linking_unknown_filetype_error(self):
        expected_err = "Don't know how to link file with extension .cuh"
        with self.assertRaisesRegex(RuntimeError, expected_err):
            @cuda.jit('void()', link=['header.cuh'])
            def kernel():
                pass

    def test_linking_file_with_no_extension_error(self):
        expected_err = "Don't know how to link file with no extension"
        with self.assertRaisesRegex(RuntimeError, expected_err):
            @cuda.jit('void()', link=['data'])
            def kernel():
                pass

    @skip_if_cuda_includes_missing
    @skip_unless_cuda_python('NVIDIA Binding needed for NVRTC')
    def test_linking_cu_cuda_include(self):
        link = os.path.join(os.path.dirname(__file__), 'data',
                            'cuda_include.cu')

        # An exception will be raised when linking this kernel due to the
        # compile failure if CUDA includes cannot be found by Nvrtc.
        @cuda.jit('void()', link=[link])
        def kernel():
            pass

    def test_try_to_link_nonexistent(self):
        with self.assertRaises(LinkerError) as e:
            @cuda.jit('void(int32[::1])', link=['nonexistent.a'])
            def f(x):
                x[0] = 0
        self.assertIn('nonexistent.a not found', e.exception.args)

    def test_set_registers_no_max(self):
        """Ensure that the jitted kernel used in the test_set_registers_* tests
        uses more than 57 registers - this ensures that test_set_registers_*
        are really checking that they reduced the number of registers used from
        something greater than the maximum."""
        compiled = cuda.jit(func_with_lots_of_registers)
        compiled = compiled.specialize(np.empty(32), *range(6))
        self.assertGreater(compiled.get_regs_per_thread(), 57)

    def test_set_registers_57(self):
        compiled = cuda.jit(max_registers=57)(func_with_lots_of_registers)
        compiled = compiled.specialize(np.empty(32), *range(6))
        self.assertLessEqual(compiled.get_regs_per_thread(), 57)

    def test_set_registers_38(self):
        compiled = cuda.jit(max_registers=38)(func_with_lots_of_registers)
        compiled = compiled.specialize(np.empty(32), *range(6))
        self.assertLessEqual(compiled.get_regs_per_thread(), 38)

    def test_set_registers_eager(self):
        sig = void(float64[::1], int64, int64, int64, int64, int64, int64)
        compiled = cuda.jit(sig, max_registers=38)(func_with_lots_of_registers)
        self.assertLessEqual(compiled.get_regs_per_thread(), 38)


if __name__ == '__main__':
    unittest.main()
