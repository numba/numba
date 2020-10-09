import numpy as np
from collections import namedtuple

from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase


class TestCudaArrayArg(CUDATestCase):
    def test_array_ary(self):

        @cuda.jit('double(double[:],int64)', device=True, inline=True)
        def device_function(a, c):
            return a[c]

        @cuda.jit('void(double[:],double[:])')
        def kernel(x, y):
            i = cuda.grid(1)
            y[i] = device_function(x, i)

        x = np.arange(10, dtype=np.double)
        y = np.zeros_like(x)
        kernel[10, 1](x, y)
        self.assertTrue(np.all(x == y))

    def test_unituple(self):
        @cuda.jit
        def f(r, x):
            r[0] = x[0]
            r[1] = x[1]
            r[2] = x[2]

        x = (1, 2, 3)
        r = np.zeros(len(x), dtype=np.int64)
        f[1, 1](r, x)

        for i in range(len(x)):
            self.assertEqual(r[i], x[i])

    def test_tuple(self):
        @cuda.jit
        def f(r1, r2, x):
            r1[0] = x[0]
            r1[1] = x[1]
            r1[2] = x[2]
            r2[0] = x[3]
            r2[1] = x[4]
            r2[2] = x[5]

        x = (1, 2, 3, 4.5, 5.5, 6.5)
        r1 = np.zeros(len(x) // 2, dtype=np.int64)
        r2 = np.zeros(len(x) // 2, dtype=np.float64)
        f[1, 1](r1, r2, x)

        for i in range(len(r1)):
            self.assertEqual(r1[i], x[i])

        for i in range(len(r2)):
            self.assertEqual(r2[i], x[i + len(r1)])

    def test_namedunituple(self):
        @cuda.jit
        def f(r, x):
            r[0] = x.x
            r[1] = x.y

        Point = namedtuple('Point', ('x', 'y'))
        x = Point(1, 2)
        r = np.zeros(len(x), dtype=np.int64)
        f[1, 1](r, x)

        self.assertEqual(r[0], x.x)
        self.assertEqual(r[1], x.y)

    def test_namedtuple(self):
        @cuda.jit
        def f(r1, r2, x):
            r1[0] = x.x
            r1[1] = x.y
            r2[0] = x.r

        Point = namedtuple('Point', ('x', 'y', 'r'))
        x = Point(1, 2, 2.236)
        r1 = np.zeros(2, dtype=np.int64)
        r2 = np.zeros(1, dtype=np.float64)
        f[1, 1](r1, r2, x)

        self.assertEqual(r1[0], x.x)
        self.assertEqual(r1[1], x.y)
        self.assertEqual(r2[0], x.r)

    def test_empty_tuple(self):
        @cuda.jit
        def f(r, x):
            r[0] = len(x)

        x = tuple()
        r = np.ones(1, dtype=np.int64)
        f[1, 1](r, x)

        self.assertEqual(r[0], 0)


if __name__ == '__main__':
    unittest.main()
