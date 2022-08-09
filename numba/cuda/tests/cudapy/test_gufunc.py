import numpy as np
import numpy.core.umath_tests as ut

from collections import namedtuple
from numba import void, float32, float64
from numba import guvectorize
from numba import cuda
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
import warnings
from numba.core.errors import NumbaPerformanceWarning
from numba.tests.support import override_config


def _get_matmulcore_gufunc(dtype=float32, max_blocksize=None):
    @guvectorize([void(dtype[:, :], dtype[:, :], dtype[:, :])],
                 '(m,n),(n,p)->(m,p)',
                 target='cuda')
    def matmulcore(A, B, C):
        m, n = A.shape
        n, p = B.shape
        for i in range(m):
            for j in range(p):
                C[i, j] = 0
                for k in range(n):
                    C[i, j] += A[i, k] * B[k, j]

    gufunc = matmulcore
    if max_blocksize:
        gufunc.max_blocksize = max_blocksize
    return gufunc


@skip_on_cudasim('ufunc API unsupported in the simulator')
class TestCUDAGufunc(CUDATestCase):

    def test_gufunc_small(self):

        gufunc = _get_matmulcore_gufunc(max_blocksize=512)

        matrix_ct = 2
        A = np.arange(matrix_ct * 2 * 4, dtype=np.float32).reshape(matrix_ct, 2,
                                                                   4)
        B = np.arange(matrix_ct * 4 * 5, dtype=np.float32).reshape(matrix_ct, 4,
                                                                   5)

        C = gufunc(A, B)
        Gold = ut.matrix_multiply(A, B)
        self.assertTrue(np.allclose(C, Gold))

    def test_gufunc_auto_transfer(self):

        gufunc = _get_matmulcore_gufunc(max_blocksize=512)

        matrix_ct = 2
        A = np.arange(matrix_ct * 2 * 4, dtype=np.float32).reshape(matrix_ct, 2,
                                                                   4)
        B = np.arange(matrix_ct * 4 * 5, dtype=np.float32).reshape(matrix_ct, 4,
                                                                   5)

        dB = cuda.to_device(B)

        C = gufunc(A, dB).copy_to_host()
        Gold = ut.matrix_multiply(A, B)
        self.assertTrue(np.allclose(C, Gold))

    def test_gufunc(self):

        gufunc = _get_matmulcore_gufunc(max_blocksize=512)

        matrix_ct = 1001 # an odd number to test thread/block division in CUDA
        A = np.arange(matrix_ct * 2 * 4, dtype=np.float32).reshape(matrix_ct, 2,
                                                                   4)
        B = np.arange(matrix_ct * 4 * 5, dtype=np.float32).reshape(matrix_ct, 4,
                                                                   5)

        C = gufunc(A, B)
        Gold = ut.matrix_multiply(A, B)
        self.assertTrue(np.allclose(C, Gold))

    def test_gufunc_hidim(self):

        gufunc = _get_matmulcore_gufunc(max_blocksize=512)

        matrix_ct = 100 # an odd number to test thread/block division in CUDA
        A = np.arange(matrix_ct * 2 * 4, dtype=np.float32).reshape(4, 25, 2, 4)
        B = np.arange(matrix_ct * 4 * 5, dtype=np.float32).reshape(4, 25, 4, 5)

        C = gufunc(A, B)
        Gold = ut.matrix_multiply(A, B)
        self.assertTrue(np.allclose(C, Gold))

    def test_gufunc_new_axis(self):

        gufunc = _get_matmulcore_gufunc(dtype=float64)

        X = np.random.randn(10, 3, 3)
        Y = np.random.randn(3, 3)

        gold = ut.matrix_multiply(X, Y)

        res1 = gufunc(X, Y)
        np.testing.assert_allclose(gold, res1)

        res2 = gufunc(X, np.tile(Y, (10, 1, 1)))
        np.testing.assert_allclose(gold, res2)

    def test_gufunc_adjust_blocksize(self):

        gufunc = _get_matmulcore_gufunc(max_blocksize=512)

        matrix_ct = 1001 # an odd number to test thread/block division in CUDA
        A = np.arange(matrix_ct * 2 * 4, dtype=np.float32).reshape(matrix_ct, 2,
                                                                   4)
        B = np.arange(matrix_ct * 4 * 5, dtype=np.float32).reshape(matrix_ct, 4,
                                                                   5)

        gufunc.max_blocksize = 32
        C = gufunc(A, B)
        Gold = ut.matrix_multiply(A, B)
        self.assertTrue(np.allclose(C, Gold))

    def test_gufunc_stream(self):

        gufunc = _get_matmulcore_gufunc(max_blocksize=512)

        #cuda.driver.flush_pending_free()
        matrix_ct = 1001 # an odd number to test thread/block division in CUDA
        A = np.arange(matrix_ct * 2 * 4, dtype=np.float32).reshape(matrix_ct, 2,
                                                                   4)
        B = np.arange(matrix_ct * 4 * 5, dtype=np.float32).reshape(matrix_ct, 4,
                                                                   5)

        stream = cuda.stream()
        dA = cuda.to_device(A, stream)
        dB = cuda.to_device(B, stream)

        dC = cuda.device_array(shape=(1001, 2, 5), dtype=A.dtype, stream=stream)
        dC = gufunc(dA, dB, out=dC, stream=stream)
        C = dC.copy_to_host(stream=stream)
        stream.synchronize()

        Gold = ut.matrix_multiply(A, B)

        self.assertTrue(np.allclose(C, Gold))

    def test_copy(self):

        @guvectorize([void(float32[:], float32[:])],
                     '(x)->(x)',
                     target='cuda')
        def copy(A, B):
            for i in range(B.size):
                B[i] = A[i]

        A = np.arange(10, dtype=np.float32) + 1
        B = np.zeros_like(A)
        copy(A, out=B)
        self.assertTrue(np.allclose(A, B))

    def test_copy_odd(self):

        @guvectorize([void(float32[:], float32[:])],
                     '(x)->(x)',
                     target='cuda')
        def copy(A, B):
            for i in range(B.size):
                B[i] = A[i]

        A = np.arange(11, dtype=np.float32) + 1
        B = np.zeros_like(A)
        copy(A, out=B)
        self.assertTrue(np.allclose(A, B))

    def test_copy2d(self):

        @guvectorize([void(float32[:, :], float32[:, :])],
                     '(x, y)->(x, y)',
                     target='cuda')
        def copy2d(A, B):
            for x in range(B.shape[0]):
                for y in range(B.shape[1]):
                    B[x, y] = A[x, y]

        A = np.arange(30, dtype=np.float32).reshape(5, 6) + 1
        B = np.zeros_like(A)
        copy2d(A, out=B)
        self.assertTrue(np.allclose(A, B))

    # Test inefficient use of the GPU where the inputs are all mapped onto a
    # single thread in a single block.
    def test_inefficient_launch_configuration(self):
        @guvectorize(['void(float32[:], float32[:], float32[:])'],
                     '(n),(n)->(n)', target='cuda')
        def numba_dist_cuda(a, b, dist):
            len = a.shape[0]
            for i in range(len):
                dist[i] = a[i] * b[i]

        a = np.random.rand(1024 * 32).astype('float32')
        b = np.random.rand(1024 * 32).astype('float32')
        dist = np.zeros(a.shape[0]).astype('float32')

        with override_config('CUDA_LOW_OCCUPANCY_WARNINGS', 1):
            with warnings.catch_warnings(record=True) as w:
                numba_dist_cuda(a, b, dist)
                self.assertEqual(w[0].category, NumbaPerformanceWarning)
                self.assertIn('Grid size', str(w[0].message))
                self.assertIn('low occupancy', str(w[0].message))

    def test_efficient_launch_configuration(self):
        @guvectorize(['void(float32[:], float32[:], float32[:])'],
                     '(n),(n)->(n)', nopython=True, target='cuda')
        def numba_dist_cuda2(a, b, dist):
            len = a.shape[0]
            for i in range(len):
                dist[i] = a[i] * b[i]

        a = np.random.rand(524288 * 2).astype('float32').\
            reshape((524288, 2))
        b = np.random.rand(524288 * 2).astype('float32').\
            reshape((524288, 2))
        dist = np.zeros_like(a)

        with override_config('CUDA_LOW_OCCUPANCY_WARNINGS', 1):
            with warnings.catch_warnings(record=True) as w:
                numba_dist_cuda2(a, b, dist)
                self.assertEqual(len(w), 0)

    def test_nopython_flag(self):

        def foo(A, B):
            pass

        # nopython = True is fine
        guvectorize([void(float32[:], float32[:])], '(x)->(x)', target='cuda',
                    nopython=True)(foo)

        # nopython = False is bad
        with self.assertRaises(TypeError) as raises:
            guvectorize([void(float32[:], float32[:])], '(x)->(x)',
                        target='cuda', nopython=False)(foo)
        self.assertEqual("nopython flag must be True", str(raises.exception))

    def test_invalid_flags(self):
        # Check invalid flags
        def foo(A, B):
            pass

        with self.assertRaises(TypeError) as raises:
            guvectorize([void(float32[:], float32[:])], '(x)->(x)',
                        target='cuda', what1=True, ever2=False)(foo)
        head = "The following target options are not supported:"
        msg = str(raises.exception)
        self.assertEqual(msg[:len(head)], head)
        items = msg[len(head):].strip().split(',')
        items = [i.strip("'\" ") for i in items]
        self.assertEqual(set(['what1', 'ever2']), set(items))

    def test_duplicated_output(self):
        @guvectorize([void(float32[:], float32[:])], '(x)->(x)', target='cuda')
        def foo(inp, out):
            pass  # intentionally empty; never executed

        inp = out = np.zeros(10, dtype=np.float32)
        with self.assertRaises(ValueError) as raises:
            foo(inp, out, out=out)

        msg = "cannot specify 'out' as both a positional and keyword argument"
        self.assertEqual(str(raises.exception), msg)

    def check_tuple_arg(self, a, b):
        @guvectorize([(float64[:], float64[:], float64[:])], '(n),(n)->()',
                     target='cuda')
        def gu_reduce(x, y, r):
            s = 0
            for i in range(len(x)):
                s += x[i] * y[i]
            r[0] = s

        r = gu_reduce(a, b)
        expected = np.sum(np.asarray(a) * np.asarray(b), axis=1)
        np.testing.assert_equal(expected, r)

    def test_tuple_of_tuple_arg(self):
        a = ((1.0, 2.0, 3.0),
             (4.0, 5.0, 6.0))
        b = ((1.5, 2.5, 3.5),
             (4.5, 5.5, 6.5))
        self.check_tuple_arg(a, b)

    def test_tuple_of_namedtuple_arg(self):
        Point = namedtuple('Point', ('x', 'y', 'z'))
        a = (Point(x=1.0, y=2.0, z=3.0),
             Point(x=4.0, y=5.0, z=6.0))
        b = (Point(x=1.5, y=2.5, z=3.5),
             Point(x=4.5, y=5.5, z=6.5))
        self.check_tuple_arg(a, b)

    def test_tuple_of_array_arg(self):
        a = (np.asarray((1.0, 2.0, 3.0)),
             np.asarray((4.0, 5.0, 6.0)))
        b = (np.asarray((1.5, 2.5, 3.5)),
             np.asarray((4.5, 5.5, 6.5)))
        self.check_tuple_arg(a, b)


if __name__ == '__main__':
    unittest.main()
