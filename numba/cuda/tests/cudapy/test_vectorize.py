from __future__ import print_function, absolute_import

import numpy as np

from numba import vectorize
from numba import cuda, int32, float32, float64
from numba import unittest_support as unittest
from numba.cuda.testing import skip_on_cudasim
from numba.cuda.testing import CUDATestCase
from numba import config

sig = [int32(int32, int32),
       float32(float32, float32),
       float64(float64, float64)]


target='cuda'
if config.ENABLE_CUDASIM:
    target='cpu'


test_dtypes = np.float32, np.int32


@skip_on_cudasim('ufunc API unsupported in the simulator')
class TestCUDAVectorize(CUDATestCase):
    N = 1000001

    def test_scalar(self):

        @vectorize(sig, target=target)
        def vector_add(a, b):
            return a + b

        a = 1.2
        b = 2.3
        c = vector_add(a, b)
        self.assertEqual(c, a + b)

    def test_1d(self):

        @vectorize(sig, target=target)
        def vector_add(a, b):
            return a + b

        cuda_ufunc = vector_add

        # build python ufunc
        np_ufunc = np.add

        # test it out
        def test(ty):
            data = np.array(np.random.random(self.N), dtype=ty)

            result = cuda_ufunc(data, data)
            gold = np_ufunc(data, data)
            self.assertTrue(np.allclose(gold, result), (gold, result))

        test(np.double)
        test(np.float32)
        test(np.int32)

    def test_1d_async(self):

        @vectorize(sig, target=target)
        def vector_add(a, b):
            return a + b

        cuda_ufunc = vector_add

        # build python ufunc
        np_ufunc = np.add

        # test it out
        def test(ty):
            data = np.array(np.random.random(self.N), dtype=ty)

            stream = cuda.stream()
            device_data = cuda.to_device(data, stream)
            dresult = cuda_ufunc(device_data, device_data, stream=stream)
            result = dresult.copy_to_host()
            stream.synchronize()

            gold = np_ufunc(data, data)

            self.assertTrue(np.allclose(gold, result), (gold, result))

        test(np.double)
        test(np.float32)
        test(np.int32)

    def test_nd(self):

        @vectorize(sig, target=target)
        def vector_add(a, b):
            return a + b

        cuda_ufunc = vector_add

        def test(dtype, order, nd, size=4):
            data = np.random.random((size,) * nd).astype(dtype)
            data[data != data] = 2.4
            data[data == float('inf')] = 3.8
            data[data == float('-inf')] = -3.8
            data2 = np.array(data.T, order=order)  # .copy(order=order)

            result = data + data2
            our_result = cuda_ufunc(data, data2)
            self.assertTrue(np.allclose(result, our_result),
                            (dtype, order, result, our_result))

        for nd in range(1, 8):
            for dtype in test_dtypes:
                for order in ('C', 'F'):
                    test(dtype, order, nd)

    def test_ufunc_attrib(self):
        self.reduce_test(8)
        self.reduce_test(100)
        self.reduce_test(2 ** 10 + 1)
        self.reduce_test2(8)
        self.reduce_test2(100)
        self.reduce_test2(2 ** 10 + 1)

    def test_output_arg(self):
        @vectorize(sig, target=target)
        def vector_add(a, b):
            return a + b

        A = np.arange(10, dtype=np.float32)
        B = np.arange(10, dtype=np.float32)
        C = np.empty_like(A)
        vector_add(A, B, out=C)
        self.assertTrue(np.allclose(A + B, C))

    def reduce_test(self, n):
        @vectorize(sig, target=target)
        def vector_add(a, b):
            return a + b

        cuda_ufunc = vector_add
        x = np.arange(n, dtype=np.int32)
        gold = np.add.reduce(x)
        result = cuda_ufunc.reduce(x)
        self.assertEqual(result, gold)

    def reduce_test2(self, n):

        @vectorize(sig, target=target)
        def vector_add(a, b):
            return a + b

        cuda_ufunc = vector_add

        x = np.arange(n, dtype=np.int32)
        gold = np.add.reduce(x)
        stream = cuda.stream()
        dx = cuda.to_device(x, stream)
        result = cuda_ufunc.reduce(dx, stream=stream)
        self.assertEqual(result, gold)

    def test_auto_transfer(self):
        @vectorize(sig, target=target)
        def vector_add(a, b):
            return a + b

        cuda_ufunc = vector_add

        n = 10
        x = np.arange(n, dtype=np.int32)
        dx = cuda.to_device(x)
        y = cuda_ufunc(x, dx).copy_to_host()
        np.testing.assert_equal(y, x + x)

    def test_ufunc_output_ravel(self):
        @vectorize(sig, target=target)
        def vector_add(a, b):
            return a + b

        cuda_ufunc = vector_add

        n = 10
        x = np.arange(n, dtype=np.int32).reshape(2, 5)
        dx = cuda.to_device(x)
        cuda_ufunc(dx, dx, out=dx)

        got = dx.copy_to_host()
        expect = x + x
        np.testing.assert_equal(got, expect)


if __name__ == '__main__':
    unittest.main()
