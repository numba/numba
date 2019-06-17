from __future__ import print_function

import itertools

import numpy as np

import numba.unittest_support as unittest
from numba import types, jit, typeof
from numba.errors import TypingError
from .support import MemoryLeakMixin, TestCase, tag


def getitem_usecase(a, b):
    return a[b]

def setitem_usecase(a, idx, b):
    a[idx] = b

def np_take(A, indices):
    return np.take(A, indices)

def np_take_kws(A, indices, axis):
    return np.take(A, indices, axis=axis)

class TestFancyIndexing(MemoryLeakMixin, TestCase):

    def generate_advanced_indices(self, N, many=True):
        choices = [np.int16([0, N - 1, -2])]
        if many:
            choices += [np.uint16([0, 1, N - 1]),
                        np.bool_([0, 1, 1, 0])]
        return choices

    def generate_basic_index_tuples(self, N, maxdim, many=True):
        """
        Generate basic index tuples with 0 to *maxdim* items.
        """
        # Note integers can be considered advanced indices in certain
        # cases, so we avoid them here.
        # See "Combining advanced and basic indexing"
        # in http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
        if many:
            choices = [slice(None, None, None),
                       slice(1, N - 1, None),
                       slice(0, None, 2),
                       slice(N - 1, None, -2),
                       slice(-N + 1, -1, None),
                       slice(-1, -N, -2),
                       ]
        else:
            choices = [slice(0, N - 1, None),
                       slice(-1, -N, -2)]
        for ndim in range(maxdim + 1):
            for tup in itertools.product(choices, repeat=ndim):
                yield tup

    def generate_advanced_index_tuples(self, N, maxdim, many=True):
        """
        Generate advanced index tuples by generating basic index tuples
        and adding a single advanced index item.
        """
        # (Note Numba doesn't support advanced indices with more than
        #  one advanced index array at the moment)
        choices = list(self.generate_advanced_indices(N, many=many))
        for i in range(maxdim + 1):
            for tup in self.generate_basic_index_tuples(N, maxdim - 1, many):
                for adv in choices:
                    yield tup[:i] + (adv,) + tup[i:]

    def generate_advanced_index_tuples_with_ellipsis(self, N, maxdim, many=True):
        """
        Same as generate_advanced_index_tuples(), but also insert an
        ellipsis at various points.
        """
        for tup in self.generate_advanced_index_tuples(N, maxdim, many):
            for i in range(len(tup) + 1):
                yield tup[:i] + (Ellipsis,) + tup[i:]

    def check_getitem_indices(self, arr, indices):
        pyfunc = getitem_usecase
        cfunc = jit(nopython=True)(pyfunc)
        orig = arr.copy()
        orig_base = arr.base or arr

        for index in indices:
            expected = pyfunc(arr, index)
            # Sanity check: if a copy wasn't made, this wasn't advanced
            # but basic indexing, and shouldn't be tested here.
            assert expected.base is not orig_base
            got = cfunc(arr, index)
            # Note Numba may not return the same array strides and
            # contiguity as Numpy
            self.assertEqual(got.shape, expected.shape)
            self.assertEqual(got.dtype, expected.dtype)
            np.testing.assert_equal(got, expected)
            # Check a copy was *really* returned by Numba
            if got.size:
                got.fill(42)
                np.testing.assert_equal(arr, orig)

    def test_getitem_tuple(self):
        # Test many variations of advanced indexing with a tuple index
        N = 4
        ndim = 3
        arr = np.arange(N ** ndim).reshape((N,) * ndim).astype(np.int32)
        indices = self.generate_advanced_index_tuples(N, ndim)

        self.check_getitem_indices(arr, indices)

    def test_getitem_tuple_and_ellipsis(self):
        # Same, but also insert an ellipsis at a random point
        N = 4
        ndim = 3
        arr = np.arange(N ** ndim).reshape((N,) * ndim).astype(np.int32)
        indices = self.generate_advanced_index_tuples_with_ellipsis(N, ndim,
                                                                    many=False)

        self.check_getitem_indices(arr, indices)

    def test_ellipsis_getsetitem(self):
        # See https://github.com/numba/numba/issues/3225
        @jit(nopython=True)
        def foo(arr, v):
            arr[..., 0] = arr[..., 1]

        arr = np.arange(2)
        foo(arr, 1)
        self.assertEqual(arr[0], arr[1])

    @tag('important')
    def test_getitem_array(self):
        # Test advanced indexing with a single array index
        N = 4
        ndim = 3
        arr = np.arange(N ** ndim).reshape((N,) * ndim).astype(np.int32)
        indices = self.generate_advanced_indices(N)
        self.check_getitem_indices(arr, indices)

    def check_setitem_indices(self, arr, indices):
        pyfunc = setitem_usecase
        cfunc = jit(nopython=True)(pyfunc)

        for index in indices:
            src = arr[index]
            expected = np.zeros_like(arr)
            got = np.zeros_like(arr)
            pyfunc(expected, index, src)
            cfunc(got, index, src)
            # Note Numba may not return the same array strides and
            # contiguity as Numpy
            self.assertEqual(got.shape, expected.shape)
            self.assertEqual(got.dtype, expected.dtype)
            np.testing.assert_equal(got, expected)

    def test_setitem_tuple(self):
        # Test many variations of advanced indexing with a tuple index
        N = 4
        ndim = 3
        arr = np.arange(N ** ndim).reshape((N,) * ndim).astype(np.int32)
        indices = self.generate_advanced_index_tuples(N, ndim)
        self.check_setitem_indices(arr, indices)

    def test_setitem_tuple_and_ellipsis(self):
        # Same, but also insert an ellipsis at a random point
        N = 4
        ndim = 3
        arr = np.arange(N ** ndim).reshape((N,) * ndim).astype(np.int32)
        indices = self.generate_advanced_index_tuples_with_ellipsis(N, ndim,
                                                                    many=False)

        self.check_setitem_indices(arr, indices)

    def test_setitem_array(self):
        # Test advanced indexing with a single array index
        N = 4
        ndim = 3
        arr = np.arange(N ** ndim).reshape((N,) * ndim).astype(np.int32) + 10
        indices = self.generate_advanced_indices(N)
        self.check_setitem_indices(arr, indices)


    def test_np_take(self):
        # shorter version of array.take test in test_array_methods
        pyfunc = np_take
        cfunc = jit(nopython=True)(pyfunc)

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

        # check unsupported arg raises
        with self.assertRaises(TypingError):
            take_kws = jit(nopython=True)(np_take_kws)
            take_kws(A, 1, 1)

        # check kwarg unsupported raises
        with self.assertRaises(TypingError):
            take_kws = jit(nopython=True)(np_take_kws)
            take_kws(A, 1, axis=1)

        #exceptions leak refs
        self.disable_leak_check()


if __name__ == '__main__':
    unittest.main()
