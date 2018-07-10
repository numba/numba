"""
Test hashing of various supported types.
"""
from __future__ import print_function

import numba.unittest_support as unittest

from collections import defaultdict

import numpy as np

from numba import jit, types, utils
import numba.unittest_support as unittest
from .support import TestCase, tag, CompilationCache


def hash_usecase(x):
    return hash(x)


class BaseTest(TestCase):

    def setUp(self):
        self.cfunc = jit(nopython=True)(hash_usecase)

    def check_collection(self, values):
        cfunc = self.cfunc
        values = list(values)
        hashes = [cfunc(x) for x in values]
        for x in hashes:
            self.assertIsInstance(x, utils.INT_TYPES)

        def check_distribution(hashes):
            distinct = set(hashes)
            if len(distinct) < 0.95 * len(values):
                # Display hash collisions, for ease of debugging
                counter = defaultdict(list)
                for v, h in zip(values, hashes):
                    counter[h].append(v)
                collisions = [(h, v) for h, v in counter.items()
                              if len(v) > 1]
                collisions = "\n".join("%s: %s" % (h, v)
                                       for h, v in sorted(collisions))
                self.fail("too many hash collisions: \n%s" % collisions)

        check_distribution(hashes)

    def int_samples(self, typ=np.int64):
        for start in (0, -50, 60000, 1<<32):
            info = np.iinfo(typ)
            if not info.min <= start <= info.max:
                continue
            n = 100
            yield range(start, start + n)
            yield range(start, start + 100 * n, 100)
            yield range(start, start + 128 * n, 128)

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


class TestNumberHashing(BaseTest):
    """
    Test hashing of number types.
    """

    def check_ints(self, typ):
        def check_values(values):
            values = sorted(set(typ(x) for x in values))
            self.check_collection(values)

        for a in self.int_samples(typ):
            check_values(a)

    def check_floats(self, typ):
        for a in self.float_samples(typ):
            self.assertEqual(a.dtype, np.dtype(typ))
            self.check_collection(a)

    def check_complex(self, typ, float_ty):
        for a in self.complex_samples(typ, float_ty):
            self.assertEqual(a.dtype, np.dtype(typ))
            self.check_collection(a)

    @tag('important')
    def test_ints(self):
        self.check_ints(np.int8)
        self.check_ints(np.uint16)
        self.check_ints(np.int32)
        self.check_ints(np.uint64)

    @tag('important')
    def test_floats(self):
        self.check_floats(np.float32)
        self.check_floats(np.float64)

    @tag('important')
    def test_complex(self):
        self.check_complex(np.complex64, np.float32)
        self.check_complex(np.complex128, np.float64)

    def test_bool(self):
        self.check_collection([False, True])


class TestTupleHashing(BaseTest):
    """
    Test hashing of tuples.
    """

    def check_tuples(self, value_generator, split):
        for values in value_generator:
            tuples = [split(a) for a in values]
            self.check_collection(tuples)

    def test_homogeneous_tuples(self):
        typ = np.uint64
        def split2(i):
            """
            Split i's bits into 2 integers.
            """
            i = typ(i)
            return (i & typ(0x5555555555555555),
                    i & typ(0xaaaaaaaaaaaaaaaa),
                    )

        def split3(i):
            """
            Split i's bits into 3 integers.
            """
            i = typ(i)
            return (i & typ(0x2492492492492492),
                    i & typ(0x4924924924924924),
                    i & typ(0x9249249249249249),
                    )

        self.check_tuples(self.int_samples(), split2)
        self.check_tuples(self.int_samples(), split3)

    @tag('important')
    def test_heterogeneous_tuples(self):
        modulo = 2**63

        def split(i):
            a = i & 0x5555555555555555
            b = (i & 0xaaaaaaaa) ^ ((i >> 32) & 0xaaaaaaaa)
            return np.int64(a), np.float64(b * 0.0001)

        self.check_tuples(self.int_samples(), split)


if __name__ == "__main__":
    unittest.main()
