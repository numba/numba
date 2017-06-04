from __future__ import print_function, absolute_import

import numpy as np

from numba import ocl
import numba.unittest_support as unittest
from numba.ocl.testing import skip_on_oclsim

class TestForAll(unittest.TestCase):
    def test_forall_1(self):

        @ocl.jit
        def foo(x):
            i = ocl.grid(1)
            if i < x.size:
                x[i] += 1

        arr = np.arange(11)
        orig = arr.copy()
        foo.forall(arr.size)(arr)
        self.assertTrue(np.all(arr == orig + 1))

    def test_forall_2(self):

        @ocl.jit("void(float32, float32[:], float32[:])")
        def bar(a, x, y):
            i = ocl.grid(1)
            if i < x.size:
                y[i] = a * x[i] + y[i]

        x = np.arange(13, dtype=np.float32)
        y = np.arange(13, dtype=np.float32)
        oldy = y.copy()
        a = 1.234
        bar.forall(y.size)(a, x, y)
        self.assertTrue(np.all(y == (a * x + oldy)))


if __name__ == '__main__':
    unittest.main()

