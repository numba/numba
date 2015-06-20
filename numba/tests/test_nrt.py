from __future__ import absolute_import, division, print_function

import numpy as np
from numba import unittest_support as unittest
from numba.runtime import rtsys
from numba.config import PYVERSION


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
        del d
        self.assertEqual(Dummy.alive, 1)
        mi.acquire()
        self.assertEqual(Dummy.alive, 1)
        mi.release()
        del mi
        self.assertEqual(Dummy.alive, 0)

    def test_meminfo_refct_2(self):
        d = Dummy()
        self.assertEqual(Dummy.alive, 1)
        addr = 0xdeadcafe  # some made up location

        mi = rtsys.meminfo_new(addr, d)
        del d
        self.assertEqual(Dummy.alive, 1)
        for _ in range(100):
            mi.acquire()
        self.assertEqual(Dummy.alive, 1)
        for _ in range(100):
            mi.release()
        del mi
        self.assertEqual(Dummy.alive, 0)

    def test_defer_dtor(self):
        d = Dummy()
        self.assertEqual(Dummy.alive, 1)
        addr = 0xdeadcafe  # some made up location

        mi = rtsys.meminfo_new(addr, d)
        # Set defer flag
        mi.defer = True
        del d
        self.assertEqual(Dummy.alive, 1)
        mi.acquire()
        self.assertEqual(Dummy.alive, 1)
        mi.release()
        del mi
        # mi refct is zero but not yet removed due to deferring
        self.assertEqual(Dummy.alive, 1)
        rtsys.process_defer_dtor()
        self.assertEqual(Dummy.alive, 0)

    @unittest.skipIf(PYVERSION <= (2, 7), "memoryview not supported")
    def test_fake_memoryview(self):
        d = Dummy()
        self.assertEqual(Dummy.alive, 1)
        addr = 0xdeadcafe  # some made up location

        mi = rtsys.meminfo_new(addr, d)
        mview = memoryview(mi)
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

    @unittest.skipIf(PYVERSION <= (2, 7), "memoryview not supported")
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


class TestNRTIssue(unittest.TestCase):
    def test_issue_with_refct_op_pruning(self):
        """
        GitHub Issue #1244 https://github.com/numba/numba/issues/1244
        """
        from numba import njit
        import numpy as np
        import math

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


if __name__ == '__main__':
    unittest.main()
