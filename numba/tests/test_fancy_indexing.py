import itertools

import numpy as np

import unittest
from numba import njit, typeof
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import MemoryLeakMixin, TestCase, tag


class TestFancyIndexing(MemoryLeakMixin, TestCase):
    # (shape or array, indices)
    # Every case has exactly one, one-dimensional array,
    # Otherwise it's not fancy indexing
    indexing_cases = [
        # Pure integers
        ((5, 6, 7, 8, 9, 10), (0, 3, np.array([0,1,3,4,2]))),
        ((5, 6, 7, 8, 9, 10), (0, np.array([0,1,3,4,2]), 5)),
        ((5, 6, 7, 8, 9, 10), (0, -3, np.array([0,1,3,4,2]))),
        ((5, 6, 7, 8, 9, 10), (0, np.array([0,1,3,4,2]), -5)),

        # Pure Slices
        ((5, 6, 7, 8, 9, 10), (slice(0, 1), slice(4, 5), np.array([0,1,3,4,2]))),
        ((5, 6, 7, 8, 9, 10), (slice(3, 4), np.array([0,1,3,4,2]), slice(None))),

        # Slices + Integers
        ((5, 6, 7, 8, 9, 10), (slice(4, 5), 3, np.array([0,1,3,4,2]), 1)),
        ((5, 6, 7, 8, 9, 10), (3, np.array([0,1,3,4,2]), slice(None), slice(4))),

        # Ellipsis
        ((5, 6, 7, 8, 9, 10), (Ellipsis, np.array([0,1,3,4,2]))),
        ((5, 6, 7, 8, 9, 10), (np.array([0,1,3,4,2]), Ellipsis)),

        # Ellipsis + Integers
        ((5, 6, 7, 8, 9, 10), (Ellipsis, 1, np.array([0,1,3,4,2]))),
        ((5, 6, 7, 8, 9, 10), (np.array([0,1,3,4,2]), 3, Ellipsis)),

        # Ellipsis + Slices
        ((5, 6, 7, 8, 9, 10), (slice(1, 2), Ellipsis, np.array([0,1,3,4,2]))),
        ((5, 6, 7, 8, 9, 10), (np.array([0,1,3,4,2]), slice(1, 4), Ellipsis)),

        # Ellipsis + Slices + Integers
        ((5, 6, 7, 8, 9, 10), (Ellipsis, 1, np.array([0,1,3,4,2]), 3, slice(1,5))),
        ((5, 6, 7, 8, 9, 10), (np.array([0,1,3,4,2]), 3, Ellipsis, slice(1,5))),

        # Boolean Arrays
        ((5, 6, 7, 8, 9, 10), (slice(4, 5), 3,
                               np.array([True, False, True, False, True, False, False]),
                               1)),
        ((5, 6, 7, 8, 9, 10), (3, np.array([True, False, True, False, True, False]),
                               slice(None), slice(4))),

        # Pure Arrays
        ((5, 6, 7, 8, 9, 10), np.array([0,3,-2], dtype=np.int16)),
        ((5, 6, 7, 8, 9, 10), np.array([0,3,1], dtype=np.uint16)),
        ((5, 6, 7, 8, 9, 10), np.array([False,True,True,False,False])),
    ]

    def check_getitem_indices(self, arr_shape, index):
        def get_item(array, idx):
            return array[index]

        arr = np.random.random_integers(0, 10, size=arr_shape)
        get_item_numba = njit(get_item)
        orig = arr.copy()
        orig_base = arr.base or arr

        expected = get_item(arr, index)
        got = get_item_numba(arr, index)
        # Sanity check: In advanced indexing, the result is always a copy.
        assert expected.base is not orig_base

        # Note: Numba may not return the same array strides and
        # contiguity as Numpy
        self.assertEqual(got.shape, expected.shape)
        self.assertEqual(got.dtype, expected.dtype)
        np.testing.assert_equal(got, expected)

        # Check a copy was *really* returned by Numba
        if got.size:
            got.fill(42)
            np.testing.assert_equal(arr, orig)

    def check_setitem_indices(self, arr_shape, index):
        def set_item(array, idx, item):
            array[idx] = item

        arr = np.random.random_integers(0, 10, size=arr_shape)
        set_item_numba = njit(set_item)
        src = arr[index]
        expected = np.zeros_like(arr)
        got = np.zeros_like(arr)

        set_item(expected, index, src)
        set_item_numba(got, index, src)

        # Note: Numba may not return the same array strides and
        # contiguity as Numpy
        self.assertEqual(got.shape, expected.shape)
        self.assertEqual(got.dtype, expected.dtype)

        np.testing.assert_equal(got, expected)

    def test_getitem(self):
        for arr_shape, idx in self.indexing_cases:
            with self.subTest(arr_shape=arr_shape, idx=idx):
                self.check_getitem_indices(arr_shape, idx)

    def test_setitem(self):
        for arr_shape, idx in self.indexing_cases:
            with self.subTest(arr_shape=arr_shape, idx=idx):
                self.check_setitem_indices(arr_shape, idx)

    def test_fancy_errors(self):
        arr_shape = (5, 6, 7, 8, 9, 10)

        # Cases with multi-dimensional indexing array
        idx = (0, 3, np.array([[1, 2], [2, 3]]))
        with self.assertRaises(TypingError) as raises:
            self.check_getitem_indices(arr_shape, idx)
        self.assertIn(
            'Numba does not support multidimensional indices.',
            str(raises.exception)
        )

        # Cases with more than one indexing array
        idx = (0, 3, np.array([1, 2]), np.array([1, 2]))
        with self.assertRaises(TypingError) as raises:
            self.check_getitem_indices(arr_shape, idx)
        self.assertIn(
            'Numba doesn\'t support more than one non-scalar array index.',
            str(raises.exception)
        )

        # Cases with more than one indexing subspace
        # (The subspaces here are separated by slice(None))
        idx = (0, np.array([1, 2]), slice(None), 3, 4)
        with self.assertRaises(TypingError) as raises:
            self.check_getitem_indices(arr_shape, idx)
        self.assertIn(
            'Numba doesn\'t support more than one indexing subspace',
            str(raises.exception)
        )

    def test_ellipsis_getsetitem(self):
        # See https://github.com/numba/numba/issues/3225
        @njit
        def foo(arr, v):
            arr[..., 0] = arr[..., 1]

        arr = np.arange(2)
        foo(arr, 1)
        self.assertEqual(arr[0], arr[1])

    def test_np_take(self):
        def np_take(array, indices):
            return np.take(array, indices)

        # shorter version of array.take test in test_array_methods
        pyfunc = np_take
        cfunc = njit(pyfunc)

        def check(arr, ind):
            expected = pyfunc(arr, ind)
            got = cfunc(arr, ind)
            self.assertPreciseEqual(expected, got)
            if hasattr(expected, 'order'):
                self.assertEqual(expected.order == got.order)

        # need to check:
        # 1. scalar index
        # 2. 1d array index
        # 3. nd array index
        # 4. reflected list
        # 5. tuples

        test_indices = []
        test_indices.append(1)
        test_indices.append(np.array([1, 5, 1, 11, 3]))
        test_indices.append(np.array([[[1], [5]], [[1], [11]]]))
        test_indices.append([1, 5, 1, 11, 3])
        test_indices.append((1, 5, 1))
        test_indices.append(((1, 5, 1), (11, 3, 2)))

        for dt in [np.int64, np.complex128]:
            A = np.arange(12, dtype=dt).reshape((4, 3))
            for ind in test_indices:
                check(A, ind)

        #check illegal access raises
        szA = A.size
        illegal_indices = [szA, -szA - 1, np.array(szA), np.array(-szA - 1),
                           [szA], [-szA - 1]]
        for x in illegal_indices:
            with self.assertRaises(IndexError):
                cfunc(A, x) # oob raises

        # check float indexing raises
        with self.assertRaises(TypingError):
            cfunc(A, [1.7])

        def np_take_kws(array, indices, axis):
            return np.take(array, indices, axis=axis)
    
        # check unsupported arg raises
        with self.assertRaises(TypingError):
            take_kws = njit(np_take_kws)
            take_kws(A, 1, 1)

        # check kwarg unsupported raises
        with self.assertRaises(TypingError):
            take_kws = njit(np_take_kws)
            take_kws(A, 1, axis=1)

        #exceptions leak refs
        self.disable_leak_check()


if __name__ == '__main__':
    unittest.main()
