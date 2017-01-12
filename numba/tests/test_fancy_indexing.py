from __future__ import print_function

import itertools

import numpy as np

import numba.unittest_support as unittest
from numba import types, jit, typeof
from .support import MemoryLeakMixin, TestCase, tag


def getitem_usecase(a, b):
    return a[b]

def setitem_usecase(a, idx, b):
    a[idx] = b


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


if __name__ == '__main__':
    unittest.main()
