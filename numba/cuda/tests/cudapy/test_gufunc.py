import numpy as np
import numpy.core.umath_tests as ut

from numba import void, float32, float64
from numba import guvectorize
from numba import cuda
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest


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


if __name__ == '__main__':
    unittest.main()
