import os.path
import numpy as np
from numba.cuda.testing import unittest
from numba.cuda.testing import skip_on_cudasim
from numba.cuda.testing import CUDATestCase
from numba.cuda.cudadrv.driver import Linker
from numba.cuda import require_context
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
        linker = Linker()
        del linker

    @require_context
    def test_linking(self):
        global bar  # must be a global; other it is recognized as a freevar
        bar = cuda.declare_device('bar', 'int32(int32)')

        link = os.path.join(os.path.dirname(__file__), 'data', 'jitlink.ptx')

        @cuda.jit('void(int32[:], int32[:])', link=[link])
        def foo(x, y):
            i = cuda.grid(1)
            x[i] += bar(y[i])

        A = np.array([123])
        B = np.array([321])

        foo[1, 1](A, B)

        self.assertTrue(A[0] == 123 + 2 * 321)

    @require_context
    def test_set_registers_no_max(self):
        """Ensure that the jitted kernel used in the test_set_registers_* tests
        uses more than 57 registers - this ensures that test_set_registers_*
        are really checking that they reduced the number of registers used from
        something greater than the maximum."""
        compiled = cuda.jit(func_with_lots_of_registers)
        compiled = compiled.specialize(np.empty(32), *range(6))
        self.assertGreater(compiled._func.get().attrs.regs, 57)

    @require_context
    def test_set_registers_57(self):
        compiled = cuda.jit(max_registers=57)(func_with_lots_of_registers)
        compiled = compiled.specialize(np.empty(32), *range(6))
        self.assertLessEqual(compiled._func.get().attrs.regs, 57)

    @require_context
    def test_set_registers_38(self):
        compiled = cuda.jit(max_registers=38)(func_with_lots_of_registers)
        compiled = compiled.specialize(np.empty(32), *range(6))
        self.assertLessEqual(compiled._func.get().attrs.regs, 38)

    @require_context
    def test_set_registers_eager(self):
        sig = void(float64[::1], int64, int64, int64, int64, int64, int64)
        compiled = cuda.jit(sig, max_registers=38)(func_with_lots_of_registers)
        self.assertLessEqual(compiled._func.get().attrs.regs, 38)


if __name__ == '__main__':
    unittest.main()
