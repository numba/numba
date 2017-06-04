from __future__ import print_function, absolute_import, division

import numpy as np

from numba import unittest_support as unittest
from numba import ocl, float32


class TestFastMathOption(unittest.TestCase):
    def test_kernel(self):

        def foo(arr, val):
            i = ocl.grid(1)
            if i < arr.size:
                arr[i] = float32(i) / val

        fastver = ocl.jit("void(float32[:], float32)", fastmath=True)(foo)
        precver = ocl.jit("void(float32[:], float32)")(foo)

        self.assertIn('div.full.ftz.f32', fastver.ptx)
        self.assertNotIn('div.full.ftz.f32', precver.ptx)

    def test_device(self):
        # fastmath option is ignored for device function
        @ocl.jit("float32(float32, float32)", device=True)
        def foo(a, b):
            return a / b

        def bar(arr, val):
            i = ocl.grid(1)
            if i < arr.size:
                arr[i] = foo(i, val)

        fastver = ocl.jit("void(float32[:], float32)", fastmath=True)(bar)
        precver = ocl.jit("void(float32[:], float32)")(bar)

        self.assertIn('div.full.ftz.f32', fastver.ptx)
        self.assertNotIn('div.full.ftz.f32', precver.ptx)


if __name__ == '__main__':
    unittest.main()
