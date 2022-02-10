"""
Test cases adapted from numba/tests/test_enums.py
"""

import numpy as np

from numba import int8, int16, int32
from numba import cuda, vectorize
from numba.tests.support import TestCase
from numba.tests.enum_usecases import (
    Color,
    Shape,
    Planet,
    RequestError,
    IntEnumWithNegatives
)


class EnumTest(TestCase):

    pairs = [
        (Color.red, Color.red),
        (Color.red, Color.green),
        (Planet.EARTH, Planet.EARTH),
        (Planet.VENUS, Planet.MARS),
        (Shape.circle, IntEnumWithNegatives.two) # IntEnum, same value
    ]

    def test_compare(self):
        @cuda.jit
        def f(a, b, out):
            out[0] = a == b
            out[1] = a != b
            out[2] = a is b
            out[3] = a is not b

        arr = np.zeros((4,), dtype=np.bool_)
        for a, b in self.pairs:
            f[1, 1](a, b, arr)
            self.assertPreciseEqual(arr, np.array([
                a == b, a != b, a is b, a is not b
            ]))

    def test_getattr_getitem(self):
        @cuda.jit
        def f(out):
            # Lookup of a enum member on its class
            out[0] = Color.red == Color.green
            out[1] = Color['red'] == Color['green']

        arr = np.zeros((2,), dtype=np.bool_)
        f[1, 1](arr)
        self.assertPreciseEqual(arr, np.array([
            Color.red == Color.green, Color['red'] == Color['green']
        ]))

    def test_return_from_device_func(self):
        @cuda.jit
        def helper(pred):
            return Color.red if pred else Color.green

        @cuda.jit
        def f(pred, out):
            out[0] = helper(pred) == Color.red
            out[1] = helper(not pred) == Color.green

        arr = np.zeros((2,), dtype=np.bool_)
        f[1, 1](True, arr)
        self.assertPreciseEqual(arr, np.array([
            Color.red == Color.red, Color.green == Color.green
        ]))

    def test_int_coerse(self):
        @cuda.jit
        def f(out):
            # Implicit coercion of intenums to ints
            out[0] = Shape.square - RequestError.not_found
            out[1] = int32(Shape.circle > IntEnumWithNegatives.one)

        arr = np.zeros((2,), dtype=np.int32)
        f[1, 1](arr)
        self.assertPreciseEqual(arr, np.array([
            Shape.square - RequestError.not_found,
            Shape.circle > IntEnumWithNegatives.one
        ], dtype=arr.dtype))

    def test_int_cast(self):
        @cuda.jit
        def f(x, out0, out1, out2):
            # Explicit coercion of intenums to ints
            out0[0] = x > int16(RequestError.internal_error)
            out1[0] = x - int32(RequestError.not_found)
            out2[0] = x + int8(Shape.circle)

        arr0 = np.zeros((1,), dtype=np.int16)
        arr1 = np.zeros((1,), dtype=np.int32)
        arr2 = np.zeros((1,), dtype=np.int8)
        x = np.int8(10)

        f[1, 1](x, arr0, arr1, arr2)
        self.assertEqual(arr0[0], x > np.int16(RequestError.internal_error))
        self.assertEqual(arr1[0], x - np.int32(RequestError.not_found))
        self.assertEqual(arr2[0], x + np.int8(Shape.circle))

    def test_vectorize(self):
        def f(x):
            if x != RequestError.not_found:
                return RequestError['internal_error']
            else:
                return RequestError.dummy

        cuda_func = vectorize("int64(int64)", target='cuda')(f)
        arr = np.array([2, 404, 500, 404])
        expected = np.array([f(x) for x in arr])
        got = cuda_func(arr)
        self.assertPreciseEqual(expected, got)
