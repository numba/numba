from __future__ import print_function, absolute_import, division

import sys
import numpy as np
import threading
import random

from numba import unittest_support as unittest
from numba import njit
from numba import utils
from numba.numpy_support import version as numpy_version
from .support import MemoryLeakMixin, TestCase, tag


nrtjit = njit(_nrt=True, nogil=True)


class BaseTest(TestCase):

    def check_outputs(self, pyfunc, argslist, exact=True):
        cfunc = nrtjit(pyfunc)
        for args in argslist:
            expected = pyfunc(*args)
            ret = cfunc(*args)
            self.assertEqual(ret.size, expected.size)
            self.assertEqual(ret.dtype, expected.dtype)
            self.assertStridesEqual(ret, expected)
            if exact:
                np.testing.assert_equal(expected, ret)
            else:
                np.testing.assert_allclose(expected, ret)


class NrtRefCtTest(MemoryLeakMixin):
    def assert_array_nrt_refct(self, arr, expect):
        self.assertEqual(arr.base.refcount, expect)


class TestDynArray(NrtRefCtTest, TestCase):

    def test_empty_0d(self):
        @nrtjit
        def foo():
            arr = np.empty(())
            arr[()] = 42
            return arr

        arr = foo()
        self.assert_array_nrt_refct(arr, 1)
        np.testing.assert_equal(42, arr)
        self.assertEqual(arr.size, 1)
        self.assertEqual(arr.shape, ())
        self.assertEqual(arr.dtype, np.dtype(np.float64))
        self.assertEqual(arr.strides, ())
        arr.fill(123)  # test writability
        np.testing.assert_equal(123, arr)
        del arr

    def test_empty_1d(self):
        @nrtjit
        def foo(n):
            arr = np.empty(n)
            for i in range(n):
                arr[i] = i

            return arr

        n = 3
        arr = foo(n)
        self.assert_array_nrt_refct(arr, 1)
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
        self.assert_array_nrt_refct(got_arr, 1)
        np.testing.assert_equal(expected_arr, got_arr)

        self.assertEqual(expected_arr.size, got_arr.size)
        self.assertEqual(expected_arr.shape, got_arr.shape)
        self.assertEqual(expected_arr.strides, got_arr.strides)

        del got_arr

    @tag('important')
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
        self.assert_array_nrt_refct(got_arr, 1)
        np.testing.assert_equal(expected_arr, got_arr)

        self.assertEqual(expected_arr.size, got_arr.size)
        self.assertEqual(expected_arr.shape, got_arr.shape)
        self.assertEqual(expected_arr.strides, got_arr.strides)

        del got_arr

    @tag('important')
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
        self.assert_array_nrt_refct(got_arr, 1)
        np.testing.assert_equal(expected_arr, got_arr)

        self.assertEqual(expected_arr.size, got_arr.size)
        self.assertEqual(expected_arr.shape, got_arr.shape)
        self.assertEqual(expected_arr.strides, got_arr.strides)

        del got_arr

    @tag('important')
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

    @tag('important')
    def test_return_global_array_sliced(self):
        y = np.ones(4, dtype=np.float32)

        def return_external_array():
            return y[2:]

        cfunc = nrtjit(return_external_array)
        out = cfunc()
        self.assertIsNone(out.base)

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

    @tag('important')
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

        self.assert_array_nrt_refct(cfunc(arr_a, arr_b), 1)

        # 2D case
        arr_a = np.random.random(10).reshape(2, 5)
        arr_b = np.random.random(10).reshape(2, 5)

        np.testing.assert_equal(pyfunc(arr_a, arr_b),
                                cfunc(arr_a, arr_b))

        self.assert_array_nrt_refct(cfunc(arr_a, arr_b), 1)

        # 3D case
        arr_a = np.random.random(70).reshape(2, 5, 7)
        arr_b = np.random.random(70).reshape(2, 5, 7)

        np.testing.assert_equal(pyfunc(arr_a, arr_b),
                                cfunc(arr_a, arr_b))

        self.assert_array_nrt_refct(cfunc(arr_a, arr_b), 1)

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
        expect, got = pyfunc(x, y, t), cfunc(x, y, t)
        self.assertIsNone(got[0].base)
        self.assertIsNone(got[1].base)
        np.testing.assert_equal(expect, got)
        del expect, got
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

    def test_issue_with_return_leak(self):
        """
        Dispatcher returns a new reference.
        It need to workaround it for now.
        """
        @nrtjit
        def inner(out):
            return out

        def pyfunc(x):
            return inner(x)

        cfunc = nrtjit(pyfunc)

        arr = np.arange(10)
        old_refct = sys.getrefcount(arr)

        self.assertEqual(old_refct, sys.getrefcount(pyfunc(arr)))
        self.assertEqual(old_refct, sys.getrefcount(cfunc(arr)))
        self.assertEqual(old_refct, sys.getrefcount(arr))


class ConstructorBaseTest(NrtRefCtTest):

    def check_0d(self, pyfunc):
        cfunc = nrtjit(pyfunc)
        expected = pyfunc()
        ret = cfunc()
        self.assert_array_nrt_refct(ret, 1)
        self.assertEqual(ret.size, expected.size)
        self.assertEqual(ret.shape, expected.shape)
        self.assertEqual(ret.dtype, expected.dtype)
        self.assertEqual(ret.strides, expected.strides)
        self.check_result_value(ret, expected)
        # test writability
        expected = np.empty_like(ret) # np.full_like was not added until Numpy 1.8
        expected.fill(123)
        ret.fill(123)
        np.testing.assert_equal(ret, expected)

    def check_1d(self, pyfunc):
        cfunc = nrtjit(pyfunc)
        n = 3
        expected = pyfunc(n)
        ret = cfunc(n)
        self.assert_array_nrt_refct(ret, 1)
        self.assertEqual(ret.size, expected.size)
        self.assertEqual(ret.shape, expected.shape)
        self.assertEqual(ret.dtype, expected.dtype)
        self.assertEqual(ret.strides, expected.strides)
        self.check_result_value(ret, expected)
        # test writability
        expected = np.empty_like(ret) # np.full_like was not added until Numpy 1.8
        expected.fill(123)
        ret.fill(123)
        np.testing.assert_equal(ret, expected)
        # errors
        with self.assertRaises(ValueError) as cm:
            cfunc(-1)
        self.assertEqual(str(cm.exception), "negative dimensions not allowed")

    def check_2d(self, pyfunc):
        cfunc = nrtjit(pyfunc)
        m, n = 2, 3
        expected = pyfunc(m, n)
        ret = cfunc(m, n)
        self.assert_array_nrt_refct(ret, 1)
        self.assertEqual(ret.size, expected.size)
        self.assertEqual(ret.shape, expected.shape)
        self.assertEqual(ret.dtype, expected.dtype)
        self.assertEqual(ret.strides, expected.strides)
        self.check_result_value(ret, expected)
        # test writability
        expected = np.empty_like(ret)  # np.full_like was not added until Numpy 1.8
        expected.fill(123)
        ret.fill(123)
        np.testing.assert_equal(ret, expected)
        # errors
        with self.assertRaises(ValueError) as cm:
            cfunc(2, -1)
        self.assertEqual(str(cm.exception), "negative dimensions not allowed")


class TestNdZeros(ConstructorBaseTest, TestCase):

    def setUp(self):
        super(TestNdZeros, self).setUp()
        self.pyfunc = np.zeros

    def check_result_value(self, ret, expected):
        np.testing.assert_equal(ret, expected)

    def test_0d(self):
        pyfunc = self.pyfunc
        def func():
            return pyfunc(())
        self.check_0d(func)

    def test_1d(self):
        pyfunc = self.pyfunc
        def func(n):
            return pyfunc(n)
        self.check_1d(func)

    def test_1d_dtype(self):
        pyfunc = self.pyfunc
        def func(n):
            return pyfunc(n, np.int32)
        self.check_1d(func)

    def test_1d_dtype_instance(self):
        # dtype as numpy dtype, not as scalar class
        pyfunc = self.pyfunc
        _dtype = np.dtype('int32')
        def func(n):
            return pyfunc(n, _dtype)
        self.check_1d(func)

    def test_2d(self):
        pyfunc = self.pyfunc
        def func(m, n):
            return pyfunc((m, n))
        self.check_2d(func)

    @tag('important')
    def test_2d_dtype_kwarg(self):
        pyfunc = self.pyfunc
        def func(m, n):
            return pyfunc((m, n), dtype=np.complex64)
        self.check_2d(func)


class TestNdOnes(TestNdZeros):

    def setUp(self):
        super(TestNdOnes, self).setUp()
        self.pyfunc = np.ones


@unittest.skipIf(numpy_version < (1, 8), "test requires Numpy 1.8 or later")
class TestNdFull(ConstructorBaseTest, TestCase):

    def check_result_value(self, ret, expected):
        np.testing.assert_equal(ret, expected)

    def test_0d(self):
        def func():
            return np.full((), 4.5)
        self.check_0d(func)

    def test_1d(self):
        def func(n):
            return np.full(n, 4.5)
        self.check_1d(func)

    def test_1d_dtype(self):
        def func(n):
            return np.full(n, 4.5, np.bool_)
        self.check_1d(func)

    def test_1d_dtype_instance(self):
        dtype = np.dtype('bool')
        def func(n):
            return np.full(n, 4.5, dtype)
        self.check_1d(func)

    def test_2d(self):
        def func(m, n):
            return np.full((m, n), 4.5)
        self.check_2d(func)

    def test_2d_dtype_kwarg(self):
        def func(m, n):
            return np.full((m, n), 1 + 4.5j, dtype=np.complex64)
        self.check_2d(func)


class ConstructorLikeBaseTest(object):

    def mutate_array(self, arr):
        try:
            arr.fill(42)
        except (TypeError, ValueError):
            # Try something else (e.g. Numpy 1.6 with structured dtypes)
            fill_value = b'x' * arr.dtype.itemsize
            arr.fill(fill_value)

    def check_like(self, pyfunc, dtype):
        def check_arr(arr):
            expected = pyfunc(arr)
            ret = cfunc(arr)
            self.assertEqual(ret.size, expected.size)
            self.assertEqual(ret.dtype, expected.dtype)
            self.assertStridesEqual(ret, expected)
            self.check_result_value(ret, expected)
            # test writability
            self.mutate_array(ret)
            self.mutate_array(expected)
            np.testing.assert_equal(ret, expected)

        orig = np.linspace(0, 5, 6).astype(dtype)
        cfunc = nrtjit(pyfunc)

        for shape in (6, (2, 3), (1, 2, 3), (3, 1, 2), ()):
            if shape == ():
                arr = orig[-1:].reshape(())
            else:
                arr = orig.reshape(shape)
            check_arr(arr)
            # Non-contiguous array
            if arr.ndim > 0:
                check_arr(arr[::2])

        # Scalar argument => should produce a 0-d array
        check_arr(orig[0])


class TestNdEmptyLike(ConstructorLikeBaseTest, TestCase):

    def setUp(self):
        super(TestNdEmptyLike, self).setUp()
        self.pyfunc = np.empty_like

    def check_result_value(self, ret, expected):
        pass

    def test_like(self):
        pyfunc = self.pyfunc
        def func(arr):
            return pyfunc(arr)
        self.check_like(func, np.float64)

    def test_like_structured(self):
        dtype = np.dtype([('a', np.int16), ('b', np.float32)])
        pyfunc = self.pyfunc
        def func(arr):
            return pyfunc(arr)
        self.check_like(func, dtype)

    def test_like_dtype(self):
        pyfunc = self.pyfunc
        def func(arr):
            return pyfunc(arr, np.int32)
        self.check_like(func, np.float64)

    def test_like_dtype_instance(self):
        dtype = np.dtype('int32')
        pyfunc = self.pyfunc
        def func(arr):
            return pyfunc(arr, dtype)
        self.check_like(func, np.float64)

    def test_like_dtype_structured(self):
        dtype = np.dtype([('a', np.int16), ('b', np.float32)])
        pyfunc = self.pyfunc
        def func(arr):
            return pyfunc(arr, dtype)
        self.check_like(func, np.float64)

    def test_like_dtype_kwarg(self):
        pyfunc = self.pyfunc
        def func(arr):
            return pyfunc(arr, dtype=np.int32)
        self.check_like(func, np.float64)


class TestNdZerosLike(TestNdEmptyLike):

    def setUp(self):
        super(TestNdZerosLike, self).setUp()
        self.pyfunc = np.zeros_like

    def check_result_value(self, ret, expected):
        np.testing.assert_equal(ret, expected)

    def test_like_structured(self):
        super(TestNdZerosLike, self).test_like_structured()

    def test_like_dtype_structured(self):
        super(TestNdZerosLike, self).test_like_dtype_structured()


class TestNdOnesLike(TestNdZerosLike):

    def setUp(self):
        super(TestNdOnesLike, self).setUp()
        self.pyfunc = np.ones_like
        self.expected_value = 1

    # Not supported yet.

    @unittest.expectedFailure
    def test_like_structured(self):
        super(TestNdOnesLike, self).test_like_structured()

    @unittest.expectedFailure
    def test_like_dtype_structured(self):
        super(TestNdOnesLike, self).test_like_dtype_structured()


@unittest.skipIf(numpy_version < (1, 8), "test requires Numpy 1.8 or later")
class TestNdFullLike(ConstructorLikeBaseTest, TestCase):

    def check_result_value(self, ret, expected):
        np.testing.assert_equal(ret, expected)

    def test_like(self):
        def func(arr):
            return np.full_like(arr, 3.5)
        self.check_like(func, np.float64)

    # Not supported yet.
    @unittest.expectedFailure
    def test_like_structured(self):
        dtype = np.dtype([('a', np.int16), ('b', np.float32)])
        def func(arr):
            return np.full_like(arr, 4.5)
        self.check_like(func, dtype)

    def test_like_dtype(self):
        def func(arr):
            return np.full_like(arr, 4.5, np.bool_)
        self.check_like(func, np.float64)

    def test_like_dtype_instance(self):
        dtype = np.dtype('bool')
        def func(arr):
            return np.full_like(arr, 4.5, dtype)
        self.check_like(func, np.float64)

    def test_like_dtype_kwarg(self):
        def func(arr):
            return np.full_like(arr, 4.5, dtype=np.bool_)
        self.check_like(func, np.float64)


class TestNdIdentity(BaseTest):

    def check_identity(self, pyfunc):
        self.check_outputs(pyfunc, [(3,)])

    def test_identity(self):
        def func(n):
            return np.identity(n)
        self.check_identity(func)

    def test_identity_dtype(self):
        for dtype in (np.complex64, np.int16, np.bool_, np.dtype('bool')):
            def func(n):
                return np.identity(n, dtype)
            self.check_identity(func)


class TestNdEye(BaseTest):

    def test_eye_n(self):
        def func(n):
            return np.eye(n)
        self.check_outputs(func, [(1,), (3,)])

    def test_eye_n_m(self):
        def func(n, m):
            return np.eye(n, m)
        self.check_outputs(func, [(1, 2), (3, 2), (0, 3)])

    def check_eye_n_m_k(self, func):
        self.check_outputs(func, [(1, 2, 0),
                                  (3, 4, 1),
                                  (3, 4, -1),
                                  (4, 3, -2),
                                  (4, 3, -5),
                                  (4, 3, 5)])

    def test_eye_n_m_k(self):
        def func(n, m, k):
            return np.eye(n, m, k)
        self.check_eye_n_m_k(func)

    def test_eye_n_m_k_dtype(self):
        def func(n, m, k):
            return np.eye(N=n, M=m, k=k, dtype=np.int16)
        self.check_eye_n_m_k(func)

    def test_eye_n_m_k_dtype_instance(self):
        dtype = np.dtype('int16')
        def func(n, m, k):
            return np.eye(N=n, M=m, k=k, dtype=dtype)
        self.check_eye_n_m_k(func)


class TestNdDiag(TestCase):

    def setUp(self):
        v = np.array([1, 2, 3])
        hv = np.array([[1, 2, 3]])
        vv = np.transpose(hv)
        self.vectors = [v, hv, vv]
        a3x4 = np.arange(12).reshape(3, 4)
        a4x3 = np.arange(12).reshape(4, 3)
        self.matricies = [a3x4, a4x3]
        def func(q):
            return np.diag(q)
        self.py = func
        self.jit = nrtjit(func)

        def func_kwarg(q, k=0):
            return np.diag(q, k=k)
        self.py_kw = func_kwarg
        self.jit_kw = nrtjit(func_kwarg)

    def check_diag(self, pyfunc, nrtfunc, *args, **kwargs):
        expected = pyfunc(*args, **kwargs)
        computed = nrtfunc(*args, **kwargs)
        self.assertEqual(computed.size, expected.size)
        self.assertEqual(computed.dtype, expected.dtype)
        # NOTE: stride not tested as np returns a RO view, nb returns new data
        np.testing.assert_equal(expected, computed)

    # create a diag matrix from a vector
    def test_diag_vect_create(self):
        for d in self.vectors:
            self.check_diag(self.py, self.jit, d)

    # create a diag matrix from a vector at a given offset
    def test_diag_vect_create_kwarg(self):
        for k in range(-10, 10):
            for d in self.vectors:
                self.check_diag(self.py_kw, self.jit_kw, d, k=k)

    # extract the diagonal
    def test_diag_extract(self):
        for d in self.matricies:
            self.check_diag(self.py, self.jit, d)

    # extract a diagonal at a given offset
    def test_diag_extract_kwarg(self):
        for k in range(-4, 4):
            for d in self.matricies:
                self.check_diag(self.py_kw, self.jit_kw, d, k=k)

    # check error handling
    def test_error_handling(self):
        from numba.errors import TypingError
        d = np.array([[[1.]]])
        cfunc = nrtjit(self.py)

        # missing arg
        with self.assertRaises(TypeError):
            cfunc()

        # > 2d
        with self.assertRaises(TypingError):
            cfunc(d)
        with self.assertRaises(TypingError):
            dfunc = nrtjit(self.py_kw)
            dfunc(d, k=3)

class TestNdArange(BaseTest):

    def test_linspace_2(self):
        def pyfunc(n, m):
            return np.linspace(n, m)
        self.check_outputs(pyfunc,
                           [(0, 4), (1, 100), (-3.5, 2.5), (-3j, 2+3j),
                            (2, 1), (1+0.5j, 1.5j)], exact=False)

    def test_linspace_3(self):
        def pyfunc(n, m, p):
            return np.linspace(n, m, p)
        self.check_outputs(pyfunc,
                           [(0, 4, 9), (1, 4, 3), (-3.5, 2.5, 8),
                            (-3j, 2+3j, 7), (2, 1, 0),
                            (1+0.5j, 1.5j, 5), (1, 1e100, 1)],
                           exact=False)


class TestNpyEmptyKeyword(TestCase):
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
        for dtype in [np.int32, np.float32, np.complex64, np.dtype('complex64')]:
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
        for dtype in [np.int32, np.float32, np.complex64, np.dtype('complex64')]:
            self._test_with_shape_and_dtype_kw(dtype)

    def test_empty_no_args(self):
        from numba.errors import TypingError

        def pyfunc():
            return np.empty()

        cfunc = nrtjit(pyfunc)

        # Trigger the compilation
        # That will cause a TypingError due to missing shape argument
        with self.assertRaises(TypingError):
            cfunc()


class TestNpArray(MemoryLeakMixin, BaseTest):

    def test_0d(self):
        def pyfunc(arg):
            return np.array(arg)

        cfunc = nrtjit(pyfunc)
        got = cfunc(42)
        self.assertPreciseEqual(got, np.array(42, dtype=np.intp))
        got = cfunc(2.5)
        self.assertPreciseEqual(got, np.array(2.5))

    def test_0d_with_dtype(self):
        def pyfunc(arg):
            return np.array(arg, dtype=np.int16)

        self.check_outputs(pyfunc, [(42,), (3.5,)])

    def test_1d(self):
        def pyfunc(arg):
            return np.array(arg)

        cfunc = nrtjit(pyfunc)
        # A list
        got = cfunc([2, 3, 42])
        self.assertPreciseEqual(got, np.intp([2, 3, 42]))
        # A heterogenous tuple
        got = cfunc((1.0, 2.5j, 42))
        self.assertPreciseEqual(got, np.array([1.0, 2.5j, 42]))
        # An empty tuple
        got = cfunc(())
        self.assertPreciseEqual(got, np.float64(()))

    def test_1d_with_dtype(self):
        def pyfunc(arg):
            return np.array(arg, dtype=np.float32)

        self.check_outputs(pyfunc,
                           [([2, 42],),
                            ([3.5, 1.0],),
                            ((1, 3.5, 42),),
                            ((),),
                            ])

    @tag('important')
    def test_2d(self):
        def pyfunc(arg):
            return np.array(arg)

        cfunc = nrtjit(pyfunc)
        # A list of tuples
        got = cfunc([(1, 2), (3, 4)])
        self.assertPreciseEqual(got, np.intp([[1, 2], [3, 4]]))
        got = cfunc([(1, 2.5), (3, 4.5)])
        self.assertPreciseEqual(got, np.float64([[1, 2.5], [3, 4.5]]))
        # A tuple of lists
        got = cfunc(([1, 2], [3, 4]))
        self.assertPreciseEqual(got, np.intp([[1, 2], [3, 4]]))
        got = cfunc(([1, 2], [3.5, 4.5]))
        self.assertPreciseEqual(got, np.float64([[1, 2], [3.5, 4.5]]))
        # A tuple of tuples
        got = cfunc(((1.5, 2), (3.5, 4.5)))
        self.assertPreciseEqual(got, np.float64([[1.5, 2], [3.5, 4.5]]))
        got = cfunc(((), ()))
        self.assertPreciseEqual(got, np.float64(((), ())))

    def test_2d_with_dtype(self):
        def pyfunc(arg):
            return np.array(arg, dtype=np.int32)

        cfunc = nrtjit(pyfunc)
        got = cfunc([(1, 2.5), (3, 4.5)])
        self.assertPreciseEqual(got, np.int32([[1, 2], [3, 4]]))


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
