from __future__ import absolute_import, division, print_function

import math
import os
import sys

import numpy as np

from numba import unittest_support as unittest
from numba import njit
from numba.compiler import compile_isolated, Flags, types
from numba.runtime import rtsys
from .support import MemoryLeakMixin, TestCase

enable_nrt_flags = Flags()
enable_nrt_flags.set("nrt")


class Dummy(object):
    alive = 0

    def __init__(self):
        type(self).alive += 1

    def __del__(self):
        type(self).alive -= 1


class TestNrtMemInfo(unittest.TestCase):
    """
    Unitest for core MemInfo functionality
    """

    def setUp(self):
        # Reset the Dummy class
        Dummy.alive = 0

    def test_meminfo_refct_1(self):
        d = Dummy()
        self.assertEqual(Dummy.alive, 1)
        addr = 0xdeadcafe  # some made up location

        mi = rtsys.meminfo_new(addr, d)
        self.assertEqual(mi.refcount, 1)
        del d
        self.assertEqual(Dummy.alive, 1)
        mi.acquire()
        self.assertEqual(mi.refcount, 2)
        self.assertEqual(Dummy.alive, 1)
        mi.release()
        self.assertEqual(mi.refcount, 1)
        del mi
        self.assertEqual(Dummy.alive, 0)

    def test_meminfo_refct_2(self):
        d = Dummy()
        self.assertEqual(Dummy.alive, 1)
        addr = 0xdeadcafe  # some made up location

        mi = rtsys.meminfo_new(addr, d)
        self.assertEqual(mi.refcount, 1)
        del d
        self.assertEqual(Dummy.alive, 1)
        for ct in range(100):
            mi.acquire()
        self.assertEqual(mi.refcount, 1 + 100)
        self.assertEqual(Dummy.alive, 1)
        for _ in range(100):
            mi.release()
        self.assertEqual(mi.refcount, 1)
        del mi
        self.assertEqual(Dummy.alive, 0)

    @unittest.skipIf(sys.version_info < (3,), "memoryview not supported")
    def test_fake_memoryview(self):
        d = Dummy()
        self.assertEqual(Dummy.alive, 1)
        addr = 0xdeadcafe  # some made up location

        mi = rtsys.meminfo_new(addr, d)
        self.assertEqual(mi.refcount, 1)
        mview = memoryview(mi)
        self.assertEqual(mi.refcount, 1)
        self.assertEqual(addr, mi.data)
        self.assertFalse(mview.readonly)
        self.assertIs(mi, mview.obj)
        self.assertTrue(mview.c_contiguous)
        self.assertEqual(mview.itemsize, 1)
        self.assertEqual(mview.ndim, 1)
        del d
        del mi

        self.assertEqual(Dummy.alive, 1)
        del mview
        self.assertEqual(Dummy.alive, 0)

    @unittest.skipIf(sys.version_info < (3,), "memoryview not supported")
    def test_memoryview(self):
        from ctypes import c_uint32, c_void_p, POINTER, cast

        dtype = np.dtype(np.uint32)
        bytesize = dtype.itemsize * 10
        mi = rtsys.meminfo_alloc(bytesize, safe=True)
        addr = mi.data
        c_arr = cast(c_void_p(mi.data), POINTER(c_uint32 * 10))
        # Check 0xCB-filling
        for i in range(10):
            self.assertEqual(c_arr.contents[i], 0xcbcbcbcb)

        # Init array with ctypes
        for i in range(10):
            c_arr.contents[i] = i + 1
        mview = memoryview(mi)
        self.assertEqual(mview.nbytes, bytesize)
        self.assertFalse(mview.readonly)
        self.assertIs(mi, mview.obj)
        self.assertTrue(mview.c_contiguous)
        self.assertEqual(mview.itemsize, 1)
        self.assertEqual(mview.ndim, 1)
        del mi
        arr = np.ndarray(dtype=dtype, shape=mview.nbytes // dtype.itemsize,
                         buffer=mview)
        del mview
        # Modify array with NumPy
        np.testing.assert_equal(np.arange(arr.size) + 1, arr)

        arr += 1

        # Check value reflected in ctypes
        for i in range(10):
            self.assertEqual(c_arr.contents[i], i + 2)

        self.assertEqual(arr.ctypes.data, addr)
        del arr
        # At this point the memory is zero filled
        # We can't check this deterministically because the memory could be
        # consumed by another thread.

    def test_buffer(self):
        from ctypes import c_uint32, c_void_p, POINTER, cast

        dtype = np.dtype(np.uint32)
        bytesize = dtype.itemsize * 10
        mi = rtsys.meminfo_alloc(bytesize, safe=True)
        self.assertEqual(mi.refcount, 1)
        addr = mi.data
        c_arr = cast(c_void_p(addr), POINTER(c_uint32 * 10))
        # Check 0xCB-filling
        for i in range(10):
            self.assertEqual(c_arr.contents[i], 0xcbcbcbcb)

        # Init array with ctypes
        for i in range(10):
            c_arr.contents[i] = i + 1

        arr = np.ndarray(dtype=dtype, shape=bytesize // dtype.itemsize,
                         buffer=mi)
        self.assertEqual(mi.refcount, 1)
        del mi
        # Modify array with NumPy
        np.testing.assert_equal(np.arange(arr.size) + 1, arr)

        arr += 1

        # Check value reflected in ctypes
        for i in range(10):
            self.assertEqual(c_arr.contents[i], i + 2)

        self.assertEqual(arr.ctypes.data, addr)
        del arr
        # At this point the memory is zero filled
        # We can't check this deterministically because the memory could be
        # consumed by another thread.


@unittest.skipUnless(sys.version_info >= (3, 4),
                     "need Python 3.4+ for the tracemalloc module")
class TestTracemalloc(unittest.TestCase):
    """
    Test NRT-allocated memory can be tracked by tracemalloc.
    """

    def measure_memory_diff(self, func):
        import tracemalloc
        tracemalloc.start()
        try:
            before = tracemalloc.take_snapshot()
            # Keep the result and only delete it after taking a snapshot
            res = func()
            after = tracemalloc.take_snapshot()
            del res
            return after.compare_to(before, 'lineno')
        finally:
            tracemalloc.stop()

    def test_snapshot(self):
        N = 1000000
        dtype = np.int8

        @njit
        def alloc_nrt_memory():
            """
            Allocate and return a large array.
            """
            return np.empty(N, dtype)

        def keep_memory():
            return alloc_nrt_memory()

        def release_memory():
            alloc_nrt_memory()

        alloc_lineno = keep_memory.__code__.co_firstlineno + 1

        # Warmup JIT
        alloc_nrt_memory()

        # The large NRT-allocated array should appear topmost in the diff
        diff = self.measure_memory_diff(keep_memory)
        stat = diff[0]
        # There is a slight overhead, so the allocated size won't exactly be N
        self.assertGreaterEqual(stat.size, N)
        self.assertLess(stat.size, N * 1.01)
        frame = stat.traceback[0]
        self.assertEqual(os.path.basename(frame.filename), "test_nrt.py")
        self.assertEqual(frame.lineno, alloc_lineno)

        # If NRT memory is released before taking a snapshot, it shouldn't
        # appear.
        diff = self.measure_memory_diff(release_memory)
        stat = diff[0]
        # Something else appears, but nothing the magnitude of N
        self.assertLess(stat.size, N * 0.01)


class TestNRTIssue(MemoryLeakMixin, TestCase):
    def test_issue_with_refct_op_pruning(self):
        """
        GitHub Issue #1244 https://github.com/numba/numba/issues/1244
        """
        @njit
        def calculate_2D_vector_mag(vector):
            x, y = vector

            return math.sqrt(x ** 2 + y ** 2)

        @njit
        def normalize_2D_vector(vector):
            normalized_vector = np.empty(2, dtype=np.float64)

            mag = calculate_2D_vector_mag(vector)
            x, y = vector

            normalized_vector[0] = x / mag
            normalized_vector[1] = y / mag

            return normalized_vector

        @njit
        def normalize_vectors(num_vectors, vectors):
            normalized_vectors = np.empty((num_vectors, 2), dtype=np.float64)

            for i in range(num_vectors):
                vector = vectors[i]

                normalized_vector = normalize_2D_vector(vector)

                normalized_vectors[i, 0] = normalized_vector[0]
                normalized_vectors[i, 1] = normalized_vector[1]

            return normalized_vectors

        num_vectors = 10
        test_vectors = np.random.random((num_vectors, 2))
        got = normalize_vectors(num_vectors, test_vectors)
        expected = normalize_vectors.py_func(num_vectors, test_vectors)

        np.testing.assert_almost_equal(expected, got)

    def test_incref_after_cast(self):
        # Issue #1427: when casting a value before returning it, the
        # cast result should be incref'ed, not the original value.
        def f():
            return 0.0, np.zeros(1, dtype=np.int32)

        # Note the return type isn't the same as the tuple type above:
        # the first element is a complex rather than a float.
        cres = compile_isolated(f, (),
                                types.Tuple((types.complex128,
                                             types.Array(types.int32, 1, 'C')
                                             ))
                                )
        z, arr = cres.entry_point()
        self.assertPreciseEqual(z, 0j)
        self.assertPreciseEqual(arr, np.zeros(1, dtype=np.int32))

    def test_refct_pruning_issue_1511(self):
        @njit
        def f():
            a = np.ones(10, dtype=np.float64)
            b = np.ones(10, dtype=np.float64)
            return a, b[:]

        a, b = f()
        np.testing.assert_equal(a, b)
        np.testing.assert_equal(a, np.ones(10, dtype=np.float64))

    def test_refct_pruning_issue_1526(self):
        @njit
        def udt(image, x, y):
            next_loc = np.where(image == 1)

            if len(next_loc[0]) == 0:
                y_offset = 1
                x_offset = 1
            else:
                y_offset = next_loc[0][0]
                x_offset = next_loc[1][0]

            next_loc_x = (x - 1) + x_offset
            next_loc_y = (y - 1) + y_offset

            return next_loc_x, next_loc_y

        a = np.array([[1, 0, 1, 0, 1, 0, 0, 1, 0, 0]])
        expect = udt.py_func(a, 1, 6)
        got = udt(a, 1, 6)

        self.assertEqual(expect, got)


if __name__ == '__main__':
    unittest.main()
