import math
import numpy as np
import numpy.core.umath_tests as ut

from numba import unittest_support as unittest
from .base import TestCase

try:
    from xnd import xnd

    from numba import vectorize
    from numba.xnd.gufunc import GuFunc
    from numba.xnd.ndtypes import Function
except ImportError:
    pass


class TestGuFunc(TestCase):
    def test_reuses_existing(self):
        f = GuFunc(lambda a: a)
        self.assertEqual(len(f.already_compiled), 0)

        f(xnd(1))
        self.assertEqual(len(f.already_compiled), 1)
        self.assertSequenceEqual(f.func.kernels, ['... * int64 -> ... * int64'])

        f(xnd(1))
        self.assertEqual(len(f.already_compiled), 1)
        self.assertSequenceEqual(f.func.kernels, ['... * int64 -> ... * int64'])

        f(xnd(1.0))
        self.assertEqual(len(f.already_compiled), 2)
        self.assertEqual(len(f.func.kernels), 2)

        self.assertSetEqual(set(f.func.kernels), {
            '... * int64 -> ... * int64',
            '... * float64 -> ... * float64'
        })


class TestGuFuncInferred(TestCase):
    @staticmethod
    def sinc(x):
        if x == 0.0:
            return 1.0
        else:
            return math.sin(x * math.pi) / (math.pi * x)

    @staticmethod
    def scaled_sinc(x, scale):
        if x == 0.0:
            return scale
        else:
            return scale * (math.sin(x * math.pi) / (math.pi * x))

    def assertVectorizeEqual(self, fn, np_arrays, xnd_arrays=None):
        if xnd_arrays is None:
            xnd_arrays = list(map(xnd.from_buffer, np_arrays))
        x = GuFunc(fn)(*xnd_arrays)
        y = vectorize(fn)(*np_arrays)
        self.assertEqual(np.asanyarray(x), y)

    def test_unary(self):
        for d in ['float64', 'float32', 'int32', 'uint32']:
            with self.subTest(d=d):
                self.assertVectorizeEqual(self.sinc, [np.arange(100, dtype=d)])

    def test_binary(self):
        a = np.arange(100, dtype=np.float64)
        self.assertVectorizeEqual(
            self.scaled_sinc,
            [a, np.uint32(3)],
            [xnd.from_buffer(a), xnd(3, type='uint32')]
        )

    def test_binary_vector(self):
        for d in ['float64', 'float32', 'int32', 'uint32']:
            with self.subTest(d=d):
                self.assertVectorizeEqual(lambda a, b: a + b, [np.arange(100, dtype=d)] * 2)

    def test_non_contiguous(self):
        x, y = np.arange(12, dtype='int32'), np.arange(6, dtype='int32')
        self.assertVectorizeEqual(
            lambda a, b: a + b,
            [x[::2], y],
            [xnd.from_buffer(x)[::2], xnd.from_buffer(y)]
        )

    def test_complex_numbers(self):
        self.assertVectorizeEqual(
            lambda a, b: a + b,
            [np.arange(12, dtype='complex64') + 1j] * 2,
            [xnd([x + 1j for x in range(12)])] * 2
        )

    def test_boolean_output(self):
        self.assertVectorizeEqual(
            lambda a, b: a == b,
            [np.arange(10, dtype='int32')] * 2
        )

    def test_many_args(self):
        a = np.arange(80, dtype='float64').reshape(8, 10)
        b = a.copy()
        c = a.copy(order='F')
        d = np.arange(16 * 20, dtype='float64').reshape(16, 20)[::2, ::2]

        a_xnd = xnd.from_buffer(a)
        b_xnd = xnd.from_buffer(b)
        c_xnd = xnd.from_buffer(c)
        d_xnd = xnd.from_buffer(np.arange(16 * 20, dtype='float64').reshape(16, 20))[::2, ::2]

        self.assertVectorizeEqual(
            lambda a, b, c, d: a + b + c + d,
            [a, b, c, d],
            [a_xnd, b_xnd, c_xnd, d_xnd]
        )

    def test_multidimensional_add(self):
        x = xnd([list(range(5)), list(range(5, 10))], dtype='int64')
        res = GuFunc(lambda a, b: a + b)(x, x)
        self.assertEqual(
            res,
            xnd([list(range(0, 10, 2)), list(range(10, 20, 2))], dtype='int64')
        )

    def test_broadcasting(self):
        f = GuFunc(lambda a, b: a + b)

        self.assertEqual(
            f(xnd(1), xnd([1, 2, 3])),
            xnd([2, 3, 4]),
            '(,) * (3,) -> (3,)'
        )

        self.assertEqual(
            f(xnd.empty('5 * 4 * int64'), xnd.empty('1 * int64')),
            xnd.empty('5 * 4 * int64'),
            '(5, 4) * (1,) -> (5, 4)'
        )
        self.assertEqual(
            f(xnd.empty('8 * 1 * 6 * 1 * int64'), xnd.empty('7 * 1 * 5 * int64')),
            xnd.empty('8 * 7 * 6 * 5 * int64'),
            '(8, 1, 6, 1) * (7, 1, 5) -> (8, 7, 6, 5)'
        )

class TestGuFuncExplicit(TestCase):

    @staticmethod
    def add_vecs(a, b, c):
        for i in range(10):
            c[i] = a[i] + b[i]


    @staticmethod
    def matmulcore(A, B, C):
        m, n = A.shape
        n, p = B.shape
        for i in range(m):
            for j in range(p):
                C[i, j] = 0
                for k in range(n):
                    C[i, j] += A[i, k] * B[k, j]

    def test_finds_add_kernel(self):
        f = GuFunc(lambda a: a)
        f.compile(Function.from_ndt('float64 -> float64'))
        self.assertEquals(f.func(xnd(1.0)), xnd(1.0))

    def test_returns_array(self):
        f = GuFunc(self.add_vecs)
        sig = '10 * int64, 10 * int64 -> 10 * int64'
        f.compile(Function.from_ndt(sig))
        self.assertSequenceEqual(f.func.kernels, [sig])
        self.assertEquals(
            f.func(
                xnd(list(range(10))),
                xnd(list(range(10, 20)))
            ),
             xnd(list(range(10, 30, 2)))
        )

    def test_matmul(self):
        f = GuFunc(self.matmulcore)
        sig = 'M * N * float32, N * P * float32 -> M * P * float32'
        f.compile(Function.from_ndt(sig))
        self.assertSequenceEqual(f.func.kernels, [sig])

        A = np.arange(2 * 4, dtype=np.float32).reshape(2, 4)
        B = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
        C = f.func(xnd.from_buffer(A), xnd.from_buffer(B))

        Gold = ut.matrix_multiply(A, B)
        self.assertEqual(C, xnd.from_buffer(Gold))

    @unittest.skip('Broadcasting gufuncs are not working yet')
    def test_matmul_broadcasting(self):
        f = GuFunc(self.matmulcore)
        sig = '... * M * N * float32, ... * N * P * float32 -> ... * M * P * float32'
        f.compile(Function.from_ndt(sig))
        self.assertSequenceEqual(f.func.kernels, [sig])

        matrix_ct = 1001
        A = np.arange(matrix_ct * 2 * 4, dtype=np.float32).reshape(matrix_ct, 2, 4)
        B = np.arange(matrix_ct * 4 * 5, dtype=np.float32).reshape(matrix_ct, 4, 5)
        C = f.func(xnd.from_buffer(A), xnd.from_buffer(B))

        Gold = ut.matrix_multiply(A, B)
        self.assertEqual(C[1], xnd.from_buffer(Gold)[1])

if __name__ == '__main__':
    unittest.main()
