from __future__ import print_function, absolute_import, division

import sys
import numpy as np
import threading
import random

from numba import unittest_support as unittest
from numba import njit
from numba import utils


nrtjit = njit(_nrt=True, nogil=True)


class TestDynArray(unittest.TestCase):
    def test_empty_1d(self):
        @nrtjit
        def foo(n):
            arr = np.empty(n)
            for i in range(n):
                arr[i] = i

            return arr

        n = 3
        arr = foo(n)
        np.testing.assert_equal(np.arange(n), arr)
        self.assertEqual(arr.size, n)
        self.assertEqual(arr.shape, (n,))
        self.assertEqual(arr.dtype, np.dtype(np.float64))
        self.assertEqual(arr.strides, (np.dtype(np.float64).itemsize,))
        arr.fill(123)  # test writability
        np.testing.assert_equal(123, arr)
        del arr

    def test_empty_2d(self):
        def pyfunc(m, n):
            arr = np.empty((m, n), np.int32)
            for i in range(m):
                for j in range(n):
                    arr[i, j] = i + j

            return arr

        cfunc = nrtjit(pyfunc)
        m = 4
        n = 3
        expected_arr = pyfunc(m, n)
        got_arr = cfunc(m, n)
        np.testing.assert_equal(expected_arr, got_arr)

        self.assertEqual(expected_arr.size, got_arr.size)
        self.assertEqual(expected_arr.shape, got_arr.shape)
        self.assertEqual(expected_arr.strides, got_arr.strides)

        del got_arr

    def test_empty_3d(self):
        def pyfunc(m, n, p):
            arr = np.empty((m, n, p), np.int32)
            for i in range(m):
                for j in range(n):
                    for k in range(p):
                        arr[i, j, k] = i + j + k

            return arr

        cfunc = nrtjit(pyfunc)
        m = 4
        n = 3
        p = 2
        expected_arr = pyfunc(m, n, p)
        got_arr = cfunc(m, n, p)
        np.testing.assert_equal(expected_arr, got_arr)

        self.assertEqual(expected_arr.size, got_arr.size)
        self.assertEqual(expected_arr.shape, got_arr.shape)
        self.assertEqual(expected_arr.strides, got_arr.strides)

        del got_arr

    def test_empty_2d_sliced(self):
        def pyfunc(m, n, p):
            arr = np.empty((m, n), np.int32)
            for i in range(m):
                for j in range(n):
                    arr[i, j] = i + j

            return arr[p]

        cfunc = nrtjit(pyfunc)
        m = 4
        n = 3
        p = 2
        expected_arr = pyfunc(m, n, p)
        got_arr = cfunc(m, n, p)
        np.testing.assert_equal(expected_arr, got_arr)

        self.assertEqual(expected_arr.size, got_arr.size)
        self.assertEqual(expected_arr.shape, got_arr.shape)
        self.assertEqual(expected_arr.strides, got_arr.strides)

        del got_arr

    def test_return_global_array(self):
        y = np.ones(4, dtype=np.float32)
        initrefct = sys.getrefcount(y)

        def return_external_array():
            return y

        cfunc = nrtjit(return_external_array)
        out = cfunc()

        # out reference by cfunc
        self.assertEqual(initrefct + 1, sys.getrefcount(y))

        np.testing.assert_equal(y, out)
        np.testing.assert_equal(y, np.ones(4, dtype=np.float32))
        np.testing.assert_equal(out, np.ones(4, dtype=np.float32))

        del out
        # out is only referenced by cfunc
        self.assertEqual(initrefct + 1, sys.getrefcount(y))

        del cfunc
        # y is no longer referenced by cfunc
        self.assertEqual(initrefct, sys.getrefcount(y))

    def test_return_global_array_sliced(self):
        y = np.ones(4, dtype=np.float32)

        def return_external_array():
            return y[2:]

        cfunc = nrtjit(return_external_array)
        out = cfunc()

        yy = y[2:]
        np.testing.assert_equal(yy, out)
        np.testing.assert_equal(yy, np.ones(2, dtype=np.float32))
        np.testing.assert_equal(out, np.ones(2, dtype=np.float32))

    def test_array_pass_through(self):
        def pyfunc(y):
            return y

        arr = np.ones(4, dtype=np.float32)

        cfunc = nrtjit(pyfunc)
        expected = cfunc(arr)
        got = pyfunc(arr)

        np.testing.assert_equal(expected, arr)
        np.testing.assert_equal(expected, got)
        self.assertIs(expected, arr)
        self.assertIs(expected, got)

    def test_array_pass_through_sliced(self):
        def pyfunc(y):
            return y[y.size // 2:]

        arr = np.ones(4, dtype=np.float32)

        initrefct = sys.getrefcount(arr)

        cfunc = nrtjit(pyfunc)
        got = cfunc(arr)
        self.assertEqual(initrefct + 1, sys.getrefcount(arr))
        expected = pyfunc(arr)
        self.assertEqual(initrefct + 2, sys.getrefcount(arr))

        np.testing.assert_equal(expected, arr[arr.size // 2])
        np.testing.assert_equal(expected, got)

        del expected
        self.assertEqual(initrefct + 1, sys.getrefcount(arr))
        del got
        self.assertEqual(initrefct, sys.getrefcount(arr))

    def test_ufunc_with_allocated_output(self):

        def pyfunc(a, b):
            out = np.empty(a.shape)
            np.add(a, b, out)
            return out

        cfunc = nrtjit(pyfunc)

        # 1D case
        arr_a = np.random.random(10)
        arr_b = np.random.random(10)

        np.testing.assert_equal(pyfunc(arr_a, arr_b),
                                cfunc(arr_a, arr_b))

        # 2D case
        arr_a = np.random.random(10).reshape(2, 5)
        arr_b = np.random.random(10).reshape(2, 5)

        np.testing.assert_equal(pyfunc(arr_a, arr_b),
                                cfunc(arr_a, arr_b))

        # 3D case
        arr_a = np.random.random(70).reshape(2, 5, 7)
        arr_b = np.random.random(70).reshape(2, 5, 7)

        np.testing.assert_equal(pyfunc(arr_a, arr_b),
                                cfunc(arr_a, arr_b))

    def test_allocation_mt(self):
        """
        This test exercises the array allocation in multithreaded usecase.
        This stress the freelist inside NRT.
        """

        def pyfunc(inp):
            out = np.empty(inp.size)

            # Zero fill
            for i in range(out.size):
                out[i] = 0

            for i in range(inp[0]):
                # Allocate inside a loop
                tmp = np.empty(inp.size)
                # Write to tmp
                for j in range(tmp.size):
                    tmp[j] = inp[j]
                # out = tmp + i
                for j in range(tmp.size):
                    out[j] += tmp[j] + i

            return out

        cfunc = nrtjit(pyfunc)
        size = 10  # small array size so that the computation is short
        arr = np.random.randint(1, 10, size)
        frozen_arr = arr.copy()

        np.testing.assert_equal(pyfunc(arr), cfunc(arr))
        # Ensure we did not modify the input
        np.testing.assert_equal(frozen_arr, arr)

        workers = []
        inputs = []
        outputs = []

        # Make wrapper to store the output
        def wrapped(inp, out):
            out[:] = cfunc(inp)

        # Create a lot of worker threads to create contention
        for i in range(100):
            arr = np.random.randint(1, 10, size)
            out = np.empty_like(arr)
            thread = threading.Thread(target=wrapped,
                                      args=(arr, out),
                                      name="worker{0}".format(i))
            workers.append(thread)
            inputs.append(arr)
            outputs.append(out)

        # Launch worker threads
        for thread in workers:
            thread.start()

        # Join worker threads
        for thread in workers:
            thread.join()

        # Check result
        for inp, out in zip(inputs, outputs):
            np.testing.assert_equal(pyfunc(inp), out)

    def test_refct_mt(self):
        """
        This test exercises the refct in multithreaded code
        """

        def pyfunc(n, inp):
            out = np.empty(inp.size)
            for i in range(out.size):
                out[i] = inp[i] + 1
            # Use swap to trigger many refct ops
            for i in range(n):
                out, inp = inp, out
            return out

        cfunc = nrtjit(pyfunc)
        size = 10
        input = np.arange(size, dtype=np.float)
        expected_refct = sys.getrefcount(input)
        swapct = random.randrange(1000)
        expected = pyfunc(swapct, input)
        np.testing.assert_equal(expected, cfunc(swapct, input))
        # The following checks can discover a reference count error
        del expected
        self.assertEqual(expected_refct, sys.getrefcount(input))

        workers = []
        outputs = []
        swapcts = []

        # Make wrapper to store the output
        def wrapped(n, input, out):
            out[:] = cfunc(n, input)

        # Create worker threads
        for i in range(100):
            out = np.empty(size)
            # All thread shares the same input
            swapct = random.randrange(1000)
            thread = threading.Thread(target=wrapped,
                                      args=(swapct, input, out),
                                      name="worker{0}".format(i))
            workers.append(thread)
            outputs.append(out)
            swapcts.append(swapct)

        # Launch worker threads
        for thread in workers:
            thread.start()

        # Join worker threads
        for thread in workers:
            thread.join()

        # Check result
        for swapct, out in zip(swapcts, outputs):
            np.testing.assert_equal(pyfunc(swapct, input), out)

        del outputs, workers
        # The following checks can discover a reference count error
        self.assertEqual(expected_refct, sys.getrefcount(input))

    def test_swap(self):

        def pyfunc(x, y, t):
            """Swap array x and y for t number of times
            """
            for i in range(t):
                x, y = y, x

            return x, y


        cfunc = nrtjit(pyfunc)

        x = np.random.random(100)
        y = np.random.random(100)

        t = 100

        initrefct = sys.getrefcount(x), sys.getrefcount(y)
        np.testing.assert_equal(pyfunc(x, y, t), cfunc(x, y, t))
        self.assertEqual(initrefct, (sys.getrefcount(x), sys.getrefcount(y)))

    def test_return_tuple_of_array(self):

        def pyfunc(x):
            y = np.empty(x.size)
            for i in range(y.size):
                y[i] = x[i] + 1
            return x, y

        cfunc = nrtjit(pyfunc)

        x = np.random.random(5)
        initrefct = sys.getrefcount(x)
        expected_x, expected_y = pyfunc(x)
        got_x, got_y = cfunc(x)
        self.assertIs(x, expected_x)
        self.assertIs(x, got_x)
        np.testing.assert_equal(expected_x, got_x)
        np.testing.assert_equal(expected_y, got_y)
        del expected_x, got_x
        self.assertEqual(initrefct, sys.getrefcount(x))

        self.assertEqual(sys.getrefcount(expected_y), sys.getrefcount(got_y))

    def test_return_tuple_of_array_created(self):

        def pyfunc(x):
            y = np.empty(x.size)
            for i in range(y.size):
                y[i] = x[i] + 1
            out = y, y
            return out

        cfunc = nrtjit(pyfunc)

        x = np.random.random(5)
        expected_x, expected_y = pyfunc(x)
        got_x, got_y = cfunc(x)
        np.testing.assert_equal(expected_x, got_x)
        np.testing.assert_equal(expected_y, got_y)
        # getrefcount owns 1, got_y owns 1
        self.assertEqual(2, sys.getrefcount(got_y))
        # getrefcount owns 1, got_y owns 1
        self.assertEqual(2, sys.getrefcount(got_y))


class TestNpyEmptyKeyword(unittest.TestCase):
    def _test_with_dtype_kw(self, dtype):
        def pyfunc(shape):
            return np.empty(shape, dtype=dtype)

        shapes = [1, 5, 9]

        cfunc = nrtjit(pyfunc)
        for s in shapes:
            expected = pyfunc(s)
            got = cfunc(s)
            self.assertEqual(expected.dtype, got.dtype)
            self.assertEqual(expected.shape, got.shape)

    def test_with_dtype_kws(self):
        for dtype in [np.int32, np.float32, np.complex64]:
            self._test_with_dtype_kw(dtype)

    def _test_with_shape_and_dtype_kw(self, dtype):
        def pyfunc(shape):
            return np.empty(shape=shape, dtype=dtype)

        shapes = [1, 5, 9]

        cfunc = nrtjit(pyfunc)
        for s in shapes:
            expected = pyfunc(s)
            got = cfunc(s)
            self.assertEqual(expected.dtype, got.dtype)
            self.assertEqual(expected.shape, got.shape)

    def test_with_shape_and_dtype_kws(self):
        for dtype in [np.int32, np.float32, np.complex64]:
            self._test_with_dtype_kw(dtype)

    def test_empty_no_args(self):
        from numba.typeinfer import TypingError

        def pyfunc():
            return np.empty()


        cfunc = nrtjit(pyfunc)

        # Trigger the compilation
        # That will cause a TypingError due to missing shape argument
        with self.assertRaises(TypingError):
            cfunc()


def benchmark_refct_speed():
    def pyfunc(x, y, t):
        """Swap array x and y for t number of times
        """
        for i in range(t):
            x, y = y, x
        return x, y

    cfunc = nrtjit(pyfunc)

    x = np.random.random(100)
    y = np.random.random(100)
    t = 10000

    def bench_pyfunc():
        pyfunc(x, y, t)

    def bench_cfunc():
        cfunc(x, y, t)

    python_time = utils.benchmark(bench_pyfunc)
    numba_time = utils.benchmark(bench_cfunc)
    print(python_time)
    print(numba_time)


if __name__ == "__main__":
    unittest.main()
