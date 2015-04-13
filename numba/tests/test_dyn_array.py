from __future__ import print_function, absolute_import, division

import sys
import numpy as np
import threading

from numba import unittest_support as unittest
from numba import njit



nrtjit = njit(nrt=True)


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

        def return_external_array():
            return y

        cfunc = nrtjit(return_external_array)
        out = cfunc()

        np.testing.assert_equal(y, out)
        np.testing.assert_equal(y, np.ones(4, dtype=np.float32))
        np.testing.assert_equal(out, np.ones(4, dtype=np.float32))

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

    def test_multithread(self):

        def pyfunc(inp):
            out = np.empty(inp.size)
            tmp = 0
            for i in range(out.size):
                out[i] = tmp
                tmp = inp[i]
            return out

        cfunc = nrtjit(pyfunc)
        size = 10**5
        arr = np.random.randint(0, 1, size)

        np.testing.assert_equal(pyfunc(arr), cfunc(arr))

        workers = []
        inputs = []
        outputs = []

        # Make wrapper to store the output
        def wrapped(inp, out):
            out[:] = cfunc(inp)

        # Create worker threads
        for i in range(10):
            arr = np.random.randint(0, 1, size)
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

    def test_multithread_stress(self):

        def pyfunc(n, t):
            out = np.empty(n)
            for i in range(out.size):
                out[i] = i

            for i in range(t):
                tmp = np.empty(n)
                for j in range(tmp.size):
                    tmp[j] = i + j
                for j in range(out.size):
                    tmp[j] += out[j]
                out = tmp   # Swap the array here

            return out


        cfunc = nrtjit(pyfunc)
        size = 1000
        repeat = 2000

        expected = pyfunc(size, repeat)
        np.testing.assert_equal(expected, cfunc(size, repeat))

        workers = []
        outputs = []

        # Make wrapper to store the output
        def wrapped(n, t, out):
            out[:] = cfunc(n, t)

        # Create worker threads
        for i in range(10):
            out = np.empty(size)
            thread = threading.Thread(target=wrapped,
                                      args=(size, repeat, out),
                                      name="worker{0}".format(i))
            workers.append(thread)
            outputs.append(out)

        # Launch worker threads
        for thread in workers:
            thread.start()

        # Join worker threads
        for thread in workers:
            thread.join()

        # Check result
        for out in outputs:
            np.testing.assert_equal(expected, out)

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



if __name__ == "__main__":
    unittest.main()
