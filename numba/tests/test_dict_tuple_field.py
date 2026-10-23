"""Regression tests for selective typed-dictionary tuple-field lookup (#9374)."""
import gc
import io
import unittest

import numpy as np
from numba import njit, types
from numba.typed import Dict
from numba.experimental import jitclass
from numba.tests.support import TestCase

KEY = types.UniTuple(types.int64, 3)
VALUE = types.Tuple((types.int64[:], types.int64[:], types.float64))


def make_dict(n=120):
    d = Dict.empty(KEY, VALUE)
    a = np.array([3, 5, 7], np.int64)
    b = np.array([13], np.int64)
    for i in range(n):
        d[(i, i % 5, i % 7)] = (a, b, float(i % 17))
    return d


@njit
def sum_third(d):
    total = 0.0
    for k in d:
        total += d[k][2]
    return total


@njit
def sum_last(d):
    total = 0.0
    for k in d:
        total += d[k][-1]
    return total


@njit
def get_zero(d, k):
    return d[k][0]


@njit
def get_one(d, k):
    return d[k][1]


@njit
def get_two(d, k):
    return d[k][2]


@njit
def tuple_escapes(d, k):
    val = d[k]
    return val[0][0] + val[2]


@njit
def overwrite_then_read(d, k):
    value = d[k]
    d[k] = (value[0], value[1], 77.0)
    return d[k][2]


@njit
def delete_after_read(d, k):
    a = d[k][0]
    del d[k]
    return a


@njit
def unoptimised_float_key(d, k):
    val = d[k]
    return val[2]


@njit
def float_key(d, k):
    return d[k][2]


class TestDictTupleField(TestCase):
    def test_rewrite_without_argument_types(self):
        # Rewrites are also invoked for intermediate inlined closure IR,
        # which does not necessarily have a dispatcher argument signature.
        from numba.core.rewrites.dict_tuple_field import RewriteDictTupleField

        class State:
            args = None

        rewrite = RewriteDictTupleField(State())
        self.assertFalse(rewrite.match(None, None, None, None))

    def test_original_reduction(self):
        d = make_dict()
        expected = sum(val[2] for val in d.values())
        self.assertEqual(sum_third(d), expected)
        self.assertEqual(sum_last(d), expected)

    def test_rewrite_is_applied(self):
        d = make_dict(6)
        sum_third(d)
        f = io.StringIO()
        sum_third.inspect_types(file=f)
        self.assertIn('global(_getitem_tuple_field:', f.getvalue())

    def test_reference_lifetime(self):
        d = make_dict()
        self.assertTrue(np.array_equal(get_zero(d, (1, 1, 1)), [3, 5, 7]))
        self.assertTrue(np.array_equal(get_one(d, (1, 1, 1)), [13]))
        v = get_zero(d, (0, 0, 0))
        del d
        gc.collect()
        np.testing.assert_array_equal(v, np.array([3, 5, 7]))

    def test_deleted_value_survives(self):
        d = make_dict()
        a = delete_after_read(d, (0, 0, 0))
        self.assertNotIn((0, 0, 0), d)
        np.testing.assert_array_equal(a, np.array([3, 5, 7]))

    def test_missing(self):
        d = make_dict()
        with self.assertRaises(KeyError):
            get_two(d, (999, 0, 0))
        d.clear()
        with self.assertRaises(KeyError):
            get_two(d, (0, 0, 0))

    def test_overwrite(self):
        d = make_dict()
        self.assertEqual(overwrite_then_read(d, (1, 1, 1)), 77.0)
        self.assertEqual(d[(1, 1, 1)][2], 77.0)

    def test_multiuse_tuple(self):
        d = make_dict()
        self.assertEqual(tuple_escapes(d, (1, 1, 1)), 4.0)
        f = io.StringIO()
        tuple_escapes.inspect_types(file=f)
        self.assertNotIn('global(_getitem_tuple_field:', f.getvalue())

    def test_resize_and_tombstones(self):
        d = make_dict(600)
        for i in range(0, 600, 2):
            del d[(i, i % 5, i % 7)]
        for i in range(600, 1000):
            d[(i, i % 5, i % 7)] = (
                np.array([1], np.int64), np.array([2], np.int64),
                float(i % 13))
        self.assertEqual(sum_third(d), sum(v[2] for v in d.values()))

    def test_float_key_edgecases(self):
        d = Dict.empty(types.float64, VALUE)
        v = (np.array([1], np.int64), np.array([2], np.int64), 3.0)
        d[0.0] = v
        d[float('nan')] = v
        for k in (0.0, -0.0, float('nan')):
            try:
                expected = unoptimised_float_key(d, k)
            except Exception as exc:
                with self.assertRaises(type(exc)):
                    float_key(d, k)
            else:
                self.assertEqual(float_key(d, k), expected)

    def test_hidden_alias_mutation(self):
        d = make_dict()
        d_type = d._dict_type

        @jitclass([('other', d_type)])
        class Writer:
            def __init__(self, other):
                self.other = other

            def __getitem__(self, k):
                old = self.other[k]
                self.other[k] = (old[0], old[1], old[2] + 10.0)
                return 0

        @njit
        def after_write(d, writer, k):
            _ = writer[k]
            return d[k][2]

        writer = Writer(d)
        self.assertEqual(after_write(d, writer, (1, 1, 1)), 11.0)

    def test_homogeneous_numeric_tuple(self):
        d = Dict.empty(types.int64, types.UniTuple(types.float64, 3))
        d[1] = (4.0, 5.0, 6.0)
        @njit
        def f(d):
            return d[1][-1]
        self.assertEqual(f(d), 6.0)

    def test_homogeneous_array_tuple(self):
        arrtype = types.float64[:]
        d = Dict.empty(types.int64, types.UniTuple(arrtype, 3))
        d[1] = (np.array([4.0]), np.array([5.0]), np.array([6.0]))
        @njit
        def f(d):
            return d[1][2]
        np.testing.assert_array_equal(f(d), np.array([6.0]))

    def test_unicode_field(self):
        value_type = types.Tuple((types.int64[:], types.unicode_type,
                                  types.float64))
        d = Dict.empty(types.unicode_type, value_type)
        d['hello'] = (np.array([3], np.int64), 'world', 1.5)
        @njit
        def f(d):
            return d['hello'][1]
        self.assertEqual(f(d), 'world')

    def test_complex_field(self):
        d = Dict.empty(types.int64,
                       types.Tuple((types.float64[:], types.int64[:],
                                    types.complex128)))
        d[1] = (np.array([2.0]), np.array([3]), 2 + 7j)
        @njit
        def f(d):
            return d[1][2]
        self.assertEqual(f(d), 2 + 7j)

    def test_randomised(self):
        rng = np.random.default_rng(718)
        for _ in range(12):
            d = make_dict(int(rng.integers(0, 100)))
            for i in range(4):
                d[(i, 0, 0)] = (
                    np.arange(int(rng.integers(0, 10)), dtype=np.int64),
                    np.ones(int(rng.integers(0, 10)), dtype=np.int64),
                    float(rng.normal()))
            self.assertAlmostEqual(sum_third(d),
                                   sum(v[2] for v in d.values()), places=10)


if __name__ == '__main__':
    unittest.main()
