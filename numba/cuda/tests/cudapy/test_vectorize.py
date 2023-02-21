import numpy as np

from collections import namedtuple
from itertools import product
from numba import vectorize
from numba import cuda, int32, float32, float64
from numba.cuda.testing import skip_on_cudasim
from numba.cuda.testing import CUDATestCase
import unittest


# Signatures to test with - these are all homogeneous in dtype, so the output
# dtype should match the input dtype - the output should not have been cast
# upwards, as reported in #8400: https://github.com/numba/numba/issues/8400
signatures = [int32(int32, int32),
              float32(float32, float32),
              float64(float64, float64)]

# The order here is chosen such that each subsequent dtype might have been
# casted to a previously-used dtype. This is unlikely to be an issue for CUDA,
# but there might be future circumstances in which it becomes relevant, perhaps
# if it supported Dynamic UFuncs, and we want to ensure that an implementation
# for a the given dtype is used rather than casting the input upwards.
dtypes = (np.float64, np.float32, np.int32)

# NumPy ndarray orders
orders = ('C', 'F')

# Input sizes corresponding to operations:
# - Less than one warp,
# - Less than one block,
# - Greater than one block (i.e. many blocks)
input_sizes = (8, 100, 2 ** 10 + 1)


@skip_on_cudasim('ufunc API unsupported in the simulator')
class TestCUDAVectorize(CUDATestCase):
    # Presumably chosen as an odd number unlikely to coincide with the total
    # thread count, and large enough to ensure a significant number of blocks
    # are used.
    N = 1000001

    def test_scalar(self):

        @vectorize(signatures, target='cuda')
        def vector_add(a, b):
            return a + b

        a = 1.2
        b = 2.3
        c = vector_add(a, b)
        self.assertEqual(c, a + b)

    def test_1d(self):

        @vectorize(signatures, target='cuda')
        def vector_add(a, b):
            return a + b

        for ty in dtypes:
            data = np.array(np.random.random(self.N), dtype=ty)
            expected = np.add(data, data)
            actual = vector_add(data, data)
            np.testing.assert_allclose(expected, actual)
            self.assertEqual(actual.dtype, ty)

    def test_1d_async(self):

        @vectorize(signatures, target='cuda')
        def vector_add(a, b):
            return a + b

        stream = cuda.stream()

        for ty in dtypes:
            data = np.array(np.random.random(self.N), dtype=ty)
            device_data = cuda.to_device(data, stream)

            dresult = vector_add(device_data, device_data, stream=stream)
            actual = dresult.copy_to_host()

            expected = np.add(data, data)

            np.testing.assert_allclose(expected, actual)
            self.assertEqual(actual.dtype, ty)

    def test_nd(self):

        @vectorize(signatures, target='cuda')
        def vector_add(a, b):
            return a + b

        for nd, dtype, order in product(range(1, 8), dtypes, orders):
            shape = (4,) * nd
            data = np.random.random(shape).astype(dtype)
            data2 = np.array(data.T, order=order)

            expected = data + data2
            actual = vector_add(data, data2)
            np.testing.assert_allclose(expected, actual)
            self.assertEqual(actual.dtype, dtype)

    def test_output_arg(self):
        @vectorize(signatures, target='cuda')
        def vector_add(a, b):
            return a + b

        A = np.arange(10, dtype=np.float32)
        B = np.arange(10, dtype=np.float32)

        expected = A + B
        actual = np.empty_like(A)
        vector_add(A, B, out=actual)

        np.testing.assert_allclose(expected, actual)
        self.assertEqual(expected.dtype, actual.dtype)

    def test_reduce(self):
        @vectorize(signatures, target='cuda')
        def vector_add(a, b):
            return a + b

        dtype = np.int32

        for n in input_sizes:
            x = np.arange(n, dtype=dtype)
            expected = np.add.reduce(x)
            actual = vector_add.reduce(x)
            np.testing.assert_allclose(expected, actual)
            # np.add.reduce is special-cased to return an int64 for any int
            # arguments, so we can't compare against its returned dtype when
            # we're checking the general reduce machinery (which just happens
            # to be using addition). Instead, compare against the input dtype.
            self.assertEqual(dtype, actual.dtype)

    def test_reduce_async(self):

        @vectorize(signatures, target='cuda')
        def vector_add(a, b):
            return a + b

        stream = cuda.stream()
        dtype = np.int32

        for n in input_sizes:
            x = np.arange(n, dtype=dtype)
            expected = np.add.reduce(x)
            dx = cuda.to_device(x, stream)
            actual = vector_add.reduce(dx, stream=stream)
            np.testing.assert_allclose(expected, actual)
            # Compare against the input dtype as in test_reduce().
            self.assertEqual(dtype, actual.dtype)

    def test_manual_transfer(self):
        @vectorize(signatures, target='cuda')
        def vector_add(a, b):
            return a + b

        n = 10
        x = np.arange(n, dtype=np.int32)
        dx = cuda.to_device(x)
        expected = x + x
        actual = vector_add(x, dx).copy_to_host()
        np.testing.assert_equal(expected, actual)
        self.assertEqual(expected.dtype, actual.dtype)

    def test_ufunc_output_2d(self):
        @vectorize(signatures, target='cuda')
        def vector_add(a, b):
            return a + b

        n = 10
        x = np.arange(n, dtype=np.int32).reshape(2, 5)
        dx = cuda.to_device(x)
        vector_add(dx, dx, out=dx)

        expected = x + x
        actual = dx.copy_to_host()
        np.testing.assert_equal(expected, actual)
        self.assertEqual(expected.dtype, actual.dtype)

    def check_tuple_arg(self, a, b):
        @vectorize(signatures, target='cuda')
        def vector_add(a, b):
            return a + b

        r = vector_add(a, b)
        np.testing.assert_equal(np.asarray(a) + np.asarray(b), r)

    def test_tuple_arg(self):
        a = (1.0, 2.0, 3.0)
        b = (4.0, 5.0, 6.0)
        self.check_tuple_arg(a, b)

    def test_namedtuple_arg(self):
        Point = namedtuple('Point', ('x', 'y', 'z'))
        a = Point(x=1.0, y=2.0, z=3.0)
        b = Point(x=4.0, y=5.0, z=6.0)
        self.check_tuple_arg(a, b)

    def test_tuple_of_array_arg(self):
        arr = np.arange(10, dtype=np.int32)
        a = (arr, arr + 1)
        b = (arr + 2, arr + 2)
        self.check_tuple_arg(a, b)

    def test_tuple_of_namedtuple_arg(self):
        Point = namedtuple('Point', ('x', 'y', 'z'))
        a = (Point(x=1.0, y=2.0, z=3.0), Point(x=1.5, y=2.5, z=3.5))
        b = (Point(x=4.0, y=5.0, z=6.0), Point(x=4.5, y=5.5, z=6.5))
        self.check_tuple_arg(a, b)

    def test_namedtuple_of_array_arg(self):
        xs1 = np.arange(10, dtype=np.int32)
        ys1 = xs1 + 2
        xs2 = np.arange(10, dtype=np.int32) * 2
        ys2 = xs2 + 1
        Points = namedtuple('Points', ('xs', 'ys'))
        a = Points(xs=xs1, ys=ys1)
        b = Points(xs=xs2, ys=ys2)
        self.check_tuple_arg(a, b)

    def test_name_attribute(self):
        @vectorize('f8(f8)', target='cuda')
        def bar(x):
            return x ** 2

        self.assertEqual(bar.__name__, 'bar')


if __name__ == '__main__':
    unittest.main()
