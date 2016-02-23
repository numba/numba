"""
Test hashing of various supported types.
"""
from __future__ import print_function

import numba.unittest_support as unittest

import sys

import numpy as np

from numba import jit, types, utils
import numba.unittest_support as unittest
from .support import TestCase, tag


@jit(nopython=True)
def hash_usecase(x):
    return hash(x)


class BaseTest(TestCase):

    def check_collection(self, values):
        cfunc = hash_usecase
        hashes = [cfunc(x) for x in values]
        for x in hashes:
            self.assertIsInstance(x, utils.INT_TYPES)

        def check_distribution(hashes):
            distinct = set(hashes)
            self.assertGreater(len(distinct), 0.95 * len(values), (distinct, values))

        check_distribution(hashes)


class TestNumberHashing(BaseTest):
    """
    Test hashing of number types.
    """

    def check_ints(self, typ):
        def check_values(values):
            values = sorted(set(typ(x) for x in values))
            self.check_collection(values)

        for start in (0, -50, 60000):
            n = 100
            check_values(range(start, start + n))
            check_values(range(start, start + 100 * n, 100))
            check_values(range(start, start + 128 * n, 128))

    def float_samples(self, typ):
        info = np.finfo(typ)

        for start in (0, 10, info.max ** 0.5, info.max / 1000.0):
            n = 100
            min_step = max(info.tiny, start * info.resolution)
            for step in (1.2, min_step ** 0.5, min_step):
                if step < min_step:
                    continue
                a = np.linspace(start, start + n * step, n)
                a = a.astype(typ)
                yield a
                yield -a
                yield a + a.mean()

        # Infs, nans, zeros
        a = typ([0.0, 0.5, -0.0, -1.0, float('inf'), -float('inf'), float('nan')])
        yield a

    def complex_samples(self, typ, float_ty):
        for real in self.float_samples(float_ty):
            for imag in self.float_samples(float_ty):
                # Ensure equal sizes
                real = real[:len(imag)]
                imag = imag[:len(real)]
                a = real + typ(1j) * imag
                yield a

    def check_floats(self, typ):
        for a in self.float_samples(typ):
            self.assertEqual(a.dtype, np.dtype(typ))
            self.check_collection(a)

    def check_complex(self, typ, float_ty):
        for a in self.complex_samples(typ, float_ty):
            self.assertEqual(a.dtype, np.dtype(typ))
            self.check_collection(a)

    def test_ints(self):
        self.check_ints(np.int8)
        self.check_ints(np.uint16)
        self.check_ints(np.int32)
        self.check_ints(np.uint64)

    def test_floats(self):
        self.check_floats(np.float32)
        self.check_floats(np.float64)

    def test_complex(self):
        self.check_complex(np.complex64, np.float32)
        self.check_complex(np.complex128, np.float64)


if __name__ == "__main__":
    unittest.main()
