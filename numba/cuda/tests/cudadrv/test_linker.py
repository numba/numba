from __future__ import print_function, absolute_import, division
import os.path
import numpy as np
from numba.cuda.testing import unittest
from numba.cuda.testing import skip_on_cudasim
from numba.cuda.testing import SerialMixin
from numba.cuda.cudadrv.driver import Linker
from numba.cuda import require_context
from numba import cuda

@skip_on_cudasim('Linking unsupported in the simulator')
class TestLinker(SerialMixin, unittest.TestCase):

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

        foo(A, B)

        self.assertTrue(A[0] == 123 + 2 * 321)


if __name__ == '__main__':
    unittest.main()
