"""
Test helper functions for distance matrix calculations.
"""
import os
import numpy as np
from numba.cuda.testing import unittest, CUDATestCase
import numba.cuda.kernels.device.helper as hp
from numpy.testing import assert_allclose, assert_equal

bits = int(os.environ["MACHINE_BITS"])

if bits == 32:
    np_float = np.float32
    np_int = np.float32
elif bits == 64:
    np_float = np.float64
    np_int = np.float64

tol = 1e-5


class TestHelperFunctions(CUDATestCase):
    def test_concatenate(self):
        vec = np.random.rand(5)
        vec2 = np.random.rand(10)
        check = np.concatenate((vec, vec2))
        out = np.zeros(len(vec) + len(vec2), dtype=np_float)
        hp.concatenate(vec, vec2, out)

        assert_allclose(out, check)
        vec[0] = 100.0
        assert_allclose(out, check)

    def test_diff(self):
        vec = np.random.rand(10)
        out = np.zeros(len(vec) - 1)
        hp.diff(vec, out)
        check = np.diff(vec)
        assert_equal(out, check)

    def test_bisect_right(self):
        a = np.random.rand(10)
        a.sort()
        v = np.random.rand(5)
        ids = np.zeros_like(v[:-1], dtype=np_int)
        hp.bisect_right(a, v[:-1], ids)
        check = np.searchsorted(a, v[:-1], side="right")
        assert_equal(ids, check)

    def test_sort_by_indices(self):
        v = np.random.rand(10)
        ids = np.arange(len(v))
        np.random.shuffle(ids)
        out = np.zeros_like(v)
        hp.sort_by_indices(v, ids, out)
        check = v[ids]
        assert_equal(out, check)

        v = np.array([0.0, 0.5528931, 1.1455898, 1.5933731])
        ids = np.array([1, 2, 3, 3, 3])
        out = np.zeros_like(ids, dtype=np_float)
        hp.sort_by_indices(v, ids, out)
        check = v[ids]
        assert_allclose(out, check)

    def test_cumsum(self):
        vec = np.random.rand(10)
        out = np.zeros_like(vec, dtype=np_float)
        hp.cumsum(vec, out)
        check = np.cumsum(vec)
        assert_allclose(out, check)

    def test_divide(self):
        v = np.random.rand(10)
        b = 4.62
        out = np.zeros_like(v, dtype=np_float)
        hp.divide(v, b, out)
        check = v / b
        assert_allclose(out, check)


if __name__ == '__main__':
    unittest.main()
