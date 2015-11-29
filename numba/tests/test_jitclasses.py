from numba.jitclass import jitclass
from numba.jitstruct import jitstruct
from numba import float32, int32
from numba.utils import OrderedDict
from numba import njit
import numpy as np
from numba import unittest_support as unittest
from .support import TestCase


class TestJitClass(TestCase):
    def test_jit_class_1(self):
        spec = OrderedDict()
        spec['x'] = float32
        spec['y'] = float32
        spec['arr'] = float32[:]

        @jitclass(spec)
        class Float2AndArray(object):
            def __init__(self, x, y, arr):
                self.x = x
                self.y = y
                self.arr = arr

            def add(self, val):
                self.x += val
                self.y += val

        spec = OrderedDict()
        spec['x'] = int32
        spec['y'] = int32

        @jitclass(spec)
        class Vector2(object):
            def __init__(self, x, y):
                self.x = x
                self.y = y

        @njit
        def bar(obj):
            return obj.x + obj.y

        @njit
        def foo(a):
            obj = Float2AndArray(1, 2, a)
            obj.add(123)

            vec = Vector2(3, 4)
            return bar(obj), bar(vec), obj.arr

        inp = np.ones(10, dtype=np.float32)
        a, b, c = foo(inp)
        assert a == 123 + 1 + 123 + 2
        assert b == 3 + 4
        np.testing.assert_equal(c, inp)

    def test_jit_class_generic(self):
        def spec(x, y, arr):
            spec = OrderedDict()
            spec['x'] = x
            spec['y'] = y
            spec['arr'] = arr
            return spec

        @jitclass(spec)
        class TwoScalarAndArray(object):
            def __init__(self, x, y, arr):
                self.x = x
                self.y = y
                self.arr = arr

            def add(self, val):
                self.x += val
                self.y += val

        @njit
        def foo():
            a = np.arange(1024)
            b = TwoScalarAndArray(2, 3, a)
            return b.arr, b.x, b.y

        a = np.arange(1024)
        arr, x, y = foo()
        np.testing.assert_equal(a, arr)
        assert x == 2
        assert y == 3

    def test_byval_struct(self):
        # A JIT-Struct is a immutable copy
        spec = OrderedDict()
        spec['x'] = float32
        spec['y'] = float32

        @jitstruct(spec)
        class Vector(object):
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def bad_method(self):
                # Error due to immutability when used
                self.x = 1

            def good_method(self, y):
                return Vector(self.x, self.y + y)

        @njit
        def foo():
            vec = Vector(1, 2)
            vec2 = vec.good_method(3)
            return vec.x, vec.y, vec2.x, vec2.y

        x1, y1, x2, y2 = foo()
        assert x1 == 1
        assert y1 == 2
        assert x2 == x1
        assert y2 == y1 + 3

    def test_byval_struct_generic(self):
        def spec(x, y):
            # In type domain
            dct = OrderedDict()
            dct['x'] = x
            dct['y'] = y
            return dct

        @jitstruct(spec)
        class Vector(object):
            # Alternatively, `spec()` can be a classmethod (not implemented, yet)
            #
            # @classmethod
            # def __type_inference__(self, x, y):
            #
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def bad_method(self):
                # Error due to immutability when used
                self.x = 1

            def good_method(self, y):
                return Vector(self.x, self.y + y * 1.0)

        @njit
        def foo():
            vec = Vector(1, 2)
            vec2 = vec.good_method(3)
            return vec.x, vec.y, vec2.x, vec2.y

        x1, y1, x2, y2 = foo()
        assert x1 == 1
        assert y1 == 2
        assert x2 == x1
        assert y2 == y1 + 3


if __name__ == '__main__':
    unittest.main()
