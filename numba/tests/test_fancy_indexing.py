import itertools

import numpy as np

import unittest
from numba import njit, typeof
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import MemoryLeakMixin, TestCase, tag


class TestFancyIndexing(MemoryLeakMixin, TestCase):
    # Every case has exactly one, one-dimensional array,
    # Otherwise it's not fancy indexing
    shape = (5, 6, 7, 8, 9, 10)
    indexing_cases = [
        # Slices + Integers
        (slice(4, 5), 3, np.array([0,1,3,4,2]), 1),
        (3, np.array([0,1,3,4,2]), slice(None), slice(4)),

        # Ellipsis + Integers
        (Ellipsis, 1, np.array([0,1,3,4,2])),
        (np.array([0,1,3,4,2]), 3, Ellipsis),

        # Ellipsis + Slices + Integers
        (Ellipsis, 1, np.array([0,1,3,4,2]), 3, slice(1,5)),
        (np.array([0,1,3,4,2]), 3, Ellipsis, slice(1,5)),

        # Boolean Arrays + Integers
        (slice(4, 5), 3,
         np.array([True, False, True, False, True, False, False]),
         1),
        (3, np.array([True, False, True, False, True, False]),
         slice(None), slice(4)),
    ]

    rng = np.random.default_rng(1)

    def generate_random_indices(self):
        N = min(self.shape)
        slice_choices = [slice(None, None, None),
            slice(1, N - 1, None),
            slice(0, None, 2),
            slice(N - 1, None, -2),
            slice(-N + 1, -1, None),
            slice(-1, -N, -2),
            slice(0, N - 1, None),
            slice(-1, -N, -2)
        ]
        integer_choices = list(np.arange(N))

        indices = []

        # Generate 20 random slice cases
        for i in range(20):
            array_idx = self.rng.integers(0, 5, size=15)
            # Randomly select 4 slices from our list
            curr_idx = self.rng.choice(slice_choices, size=4).tolist()
            # Replace one of the slice with the array index
            _array_idx = self.rng.choice(4)
            curr_idx[_array_idx] = array_idx
            indices.append(tuple(curr_idx))
        
        # Generate 20 random integer cases 
        for i in range(20):
            array_idx = self.rng.integers(0, 5, size=15)
            # Randomly select 4 integers from our list
            curr_idx = self.rng.choice(integer_choices, size=4).tolist()
            # Replace one of the slice with the array index
            _array_idx = self.rng.choice(4)
            curr_idx[_array_idx] = array_idx
            indices.append(tuple(curr_idx))

        # Generate 20 random ellipsis cases
        for i in range(20):
            array_idx = self.rng.integers(0, 5, size=15)
            # Randomly select 4 slices from our list
            curr_idx = self.rng.choice(slice_choices, size=4).tolist()
            # Generate two seperate random indices, replace one with
            # array and second with Ellipsis
            _array_idx = self.rng.choice(4, size=2, replace=False)
            curr_idx[_array_idx[0]] = array_idx
            curr_idx[_array_idx[1]] = Ellipsis
            indices.append(tuple(curr_idx))

        # Generate 20 random boolean cases
        for i in range(20):
            array_idx = self.rng.integers(0, 5, size=15)
            # Randomly select 4 slices from our list
            curr_idx = self.rng.choice(slice_choices, size=4).tolist()
            # Replace one of the slice with the boolean array index
            _array_idx = self.rng.choice(4)
            bool_arr_shape = self.shape[_array_idx]
            curr_idx[_array_idx] = np.array(
                self.rng.choice(2, size=bool_arr_shape),
                dtype=bool
            )

            indices.append(tuple(curr_idx))

        return indices

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
        got.fill(42)
        np.testing.assert_equal(arr, orig)

    def check_setitem_indices(self, arr_shape, index):
        @njit     
        def set_item(array, idx, item):
            array[idx] = item

        arr = np.random.random_integers(0, 10, size=arr_shape)
        src = arr[index]
        expected = np.zeros_like(arr)
        got = np.zeros_like(arr)

        set_item.py_func(expected, index, src)
        set_item(got, index, src)

        # Note: Numba may not return the same array strides and
        # contiguity as NumPy
        self.assertEqual(got.shape, expected.shape)
        self.assertEqual(got.dtype, expected.dtype)

        np.testing.assert_equal(got, expected)

    def test_getitem(self):
        # Cases with a combination of integers + other objects
        indices = self.indexing_cases

        # Cases with permutations of either integers or objects
        indices += self.generate_random_indices()

        for idx in indices:
            with self.subTest(idx=idx):
                self.check_getitem_indices(self.shape, idx)

    def test_setitem(self):
        # Cases with a combination of integers + other objects
        indices = self.indexing_cases

        # Cases with permutations of either integers or objects
        indices += self.generate_random_indices()

        for idx in indices:
            with self.subTest(idx=idx):
                self.check_setitem_indices(self.shape, idx)

    def test_unsupported_condition_exceptions(self):
        # Cases with multi-dimensional indexing array
        idx = (0, 3, np.array([[1, 2], [2, 3]]))
        with self.assertRaises(TypingError) as raises:
            self.check_getitem_indices(self.shape, idx)
        self.assertIn(
            'Multi-dimensional indices are not supported.',
            str(raises.exception)
        )

        # Cases with more than one indexing array
        idx = (0, 3, np.array([1, 2]), np.array([1, 2]))
        with self.assertRaises(TypingError) as raises:
            self.check_getitem_indices(self.shape, idx)
        self.assertIn(
            'Using more than one non-scalar array index is unsupported.',
            str(raises.exception)
        )

        # Cases with more than one indexing subspace
        # (The subspaces here are separated by slice(None))
        idx = (0, np.array([1, 2]), slice(None), 3, 4)
        with self.assertRaises(TypingError) as raises:
            self.check_getitem_indices(self.shape, idx)
        msg = "Using more than one indexing subspace (consecutive group " +\
                "of integer or array indices) is unsupported."
        self.assertIn(
            msg,
            str(raises.exception)
        )

    def test_setitem_0d(self):
        @njit     
        def set_item(array, idx, item):
            array[idx] = item
        # Test setitem with a 0d-array
        pyfunc = set_item.py_func
        cfunc = set_item

        inps = [
            (np.zeros(3), np.array(3.14)),
            (np.zeros(2), np.array(2)),
            (np.zeros(3, dtype=np.int64), np.array(3, dtype=np.int64)),
            (np.zeros(3, dtype=np.float64), np.array(1, dtype=np.int64)),
            (np.zeros(5, dtype='<U3'), np.array('abc')),
            (np.zeros((3,), dtype='<U3'), np.array('a')),
            (np.array(['abc','def','ghi'], dtype='<U3'),
             np.array('WXYZ', dtype='<U4')),
            (np.zeros(3, dtype=complex), np.array(2+3j, dtype=complex)),
        ]

        for x1, v in inps:
            x2 = x1.copy()
            pyfunc(x1, 0, v)
            cfunc(x2, 0, v)
            self.assertPreciseEqual(x1, x2)

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
