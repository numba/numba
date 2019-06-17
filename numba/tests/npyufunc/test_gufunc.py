from __future__ import absolute_import, print_function, division

import numpy as np
import numpy.core.umath_tests as ut

from numba import unittest_support as unittest
from numba import void, float32, jit, guvectorize
from numba.npyufunc import GUVectorize
from ..support import tag, TestCase


def matmulcore(A, B, C):
    m, n = A.shape
    n, p = B.shape
    for i in range(m):
        for j in range(p):
            C[i, j] = 0
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]


def axpy(a, x, y, out):
    out[0] = a * x  + y


class TestGUFunc(TestCase):
    target = 'cpu'

    def check_matmul_gufunc(self, gufunc):
        matrix_ct = 1001
        A = np.arange(matrix_ct * 2 * 4, dtype=np.float32).reshape(matrix_ct, 2, 4)
        B = np.arange(matrix_ct * 4 * 5, dtype=np.float32).reshape(matrix_ct, 4, 5)

        C = gufunc(A, B)
        Gold = ut.matrix_multiply(A, B)

        np.testing.assert_allclose(C, Gold, rtol=1e-5, atol=1e-8)

    @tag('important')
    def test_gufunc(self):
        gufunc = GUVectorize(matmulcore, '(m,n),(n,p)->(m,p)',
                             target=self.target)
        gufunc.add((float32[:, :], float32[:, :], float32[:, :]))
        gufunc = gufunc.build_ufunc()

        self.check_matmul_gufunc(gufunc)

    @tag('important')
    def test_guvectorize_decor(self):
        gufunc = guvectorize([void(float32[:,:], float32[:,:], float32[:,:])],
                             '(m,n),(n,p)->(m,p)',
                             target=self.target)(matmulcore)

        self.check_matmul_gufunc(gufunc)

    def test_ufunc_like(self):
        # Test problem that the stride of "scalar" gufunc argument not properly
        # handled when the actual argument is an array,
        # causing the same value (first value) being repeated.
        gufunc = GUVectorize(axpy, '(), (), () -> ()', target=self.target)
        gufunc.add('(intp, intp, intp, intp[:])')
        gufunc = gufunc.build_ufunc()

        x = np.arange(10, dtype=np.intp)
        out = gufunc(x, x, x)

        np.testing.assert_equal(out, x * x + x)


class TestGUFuncParallel(TestGUFunc):
    _numba_parallel_test_ = False
    target = 'parallel'


class TestGUVectorizeScalar(TestCase):
    """
    Nothing keeps user from out-of-bound memory access
    """
    target = 'cpu'

    @tag('important')
    def test_scalar_output(self):
        """
        Note that scalar output is a 0-dimension array that acts as
        a pointer to the output location.
        """

        @guvectorize(['void(int32[:], int32[:])'], '(n)->()',
                     target=self.target, nopython=True)
        def sum_row(inp, out):
            tmp = 0.
            for i in range(inp.shape[0]):
                tmp += inp[i]
            out[()] = tmp

        # inp is (10000, 3)
        # out is (10000)
        # The outter (leftmost) dimension must match or numpy broadcasting is performed.

        inp = np.arange(30000, dtype=np.int32).reshape(10000, 3)
        out = sum_row(inp)

        # verify result
        for i in range(inp.shape[0]):
            assert out[i] == inp[i].sum()

    @tag('important')
    def test_scalar_input(self):

        @guvectorize(['int32[:], int32[:], int32[:]'], '(n),()->(n)',
                     target=self.target, nopython=True)
        def foo(inp, n, out):
            for i in range(inp.shape[0]):
                out[i] = inp[i] * n[0]

        inp = np.arange(3 * 10, dtype=np.int32).reshape(10, 3)
        # out = np.empty_like(inp)
        out = foo(inp, 2)

        # verify result
        self.assertPreciseEqual(inp * 2, out)

    def test_scalar_input_core_type(self):
        def pyfunc(inp, n, out):
            for i in range(inp.size):
                out[i] = n * (inp[i] + 1)

        my_gufunc = guvectorize(['int32[:], int32, int32[:]'],
                                '(n),()->(n)',
                                target=self.target)(pyfunc)

        # test single core loop execution
        arr = np.arange(10).astype(np.int32)
        got = my_gufunc(arr, 2)

        expected = np.zeros_like(got)
        pyfunc(arr, 2, expected)

        np.testing.assert_equal(got, expected)

        # test multiple core loop execution
        arr = np.arange(20).astype(np.int32).reshape(10, 2)
        got = my_gufunc(arr, 2)

        expected = np.zeros_like(got)
        for ax in range(expected.shape[0]):
            pyfunc(arr[ax], 2, expected[ax])

        np.testing.assert_equal(got, expected)

    def test_scalar_input_core_type_error(self):
        with self.assertRaises(TypeError) as raises:
            @guvectorize(['int32[:], int32, int32[:]'], '(n),(n)->(n)',
                         target=self.target)
            def pyfunc(a, b, c):
                pass
        self.assertEqual("scalar type int32 given for non scalar argument #2",
                         str(raises.exception))

    def test_ndim_mismatch(self):
        with self.assertRaises(TypeError) as raises:
            @guvectorize(['int32[:], int32[:]'], '(m,n)->(n)',
                         target=self.target)
            def pyfunc(a, b):
                pass
        self.assertEqual("type and shape signature mismatch for arg #1",
                         str(raises.exception))


class TestGUVectorizeScalarParallel(TestGUVectorizeScalar):
    _numba_parallel_test_ = False
    target = 'parallel'


if __name__ == '__main__':
    unittest.main()
