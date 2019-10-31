from __future__ import print_function
from numba import unittest_support as unittest
from numba import ocl
import numpy as np


class TestOclAutojit(unittest.TestCase):
    def test_device_array(self):
        @ocl.jit
        def foo(x, y):
            i = ocl.get_global_id(0)
            y[i] = x[i]

        x = np.arange(10)
        y = np.empty_like(x)

        dx = ocl.to_device(x)
        dy = ocl.to_device(y)

        foo[10, 1](dx, dy)

        dy.copy_to_host(y)

        self.assertTrue(np.all(x == y))

    def test_device_auto_jit(self):
        @ocl.jit(device=True)
        def mapper(args):
            a, b, c = args
            return a + b + c


        @ocl.jit(device=True)
        def reducer(a, b):
            return a + b


        @ocl.jit
        def driver(A, B):
            i = ocl.get_global_id(0)
            if i < B.size:
                args = A[i], A[i] + B[i], B[i]
                B[i] = reducer(mapper(args), 1)

        A = np.arange(100, dtype=np.float32)
        B = np.arange(100, dtype=np.float32)

        Acopy = A.copy()
        Bcopy = B.copy()

        driver[1, 100](A, B)

        np.testing.assert_allclose(Acopy + Acopy + Bcopy + Bcopy + 1, B)

    def test_device_auto_jit_2(self):
        @ocl.jit(device=True)
        def inner(arg):
            return arg + 1

        @ocl.jit
        def outer(argin, argout):
            argout[0] = inner(argin[0]) + inner(2)

        a = np.zeros(1)
        b = np.zeros(1)

        stream = ocl.stream()
        d_a = ocl.to_device(a, stream)
        d_b = ocl.to_device(b, stream)

        outer[1, 1, stream](d_a, d_b)

        d_b.copy_to_host(b, stream)

        self.assertEqual(b[0], (a[0] + 1) + (2 + 1))


if __name__ == '__main__':
    unittest.main()
