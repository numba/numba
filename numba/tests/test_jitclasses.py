import numpy as np
from numba import float32, int32
from numba import njit
from numba import unittest_support as unittest
from numba.jitclass import jitclass
from numba.utils import OrderedDict
from .support import TestCase, MemoryLeakMixin


class TestJitClass(TestCase, MemoryLeakMixin):
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
        self.assertEqual(a, 123 + 1 + 123 + 2)
        self.assertEqual(b, 3 + 4)
        self.assertPreciseEqual(c, inp)

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
        self.assertEqual(x, 2)
        self.assertEqual(y, 3)

    def test_byval_struct(self):
        # A JIT-Struct is a immutable copy
        spec = OrderedDict()
        spec['x'] = float32
        spec['y'] = float32

        @jitclass(spec, immutable=True)
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
        self.assertEqual(x1, 1)
        self.assertEqual(y1, 2)
        self.assertEqual(x2, x1)
        self.assertEqual(y2, y1 + 3)

    def test_byval_struct_generic(self):
        def spec(x, y):
            # In type domain
            dct = OrderedDict()
            dct['x'] = x
            dct['y'] = y
            return dct

        @jitclass(spec, immutable=True)
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
        self.assertEqual(x1, 1)
        self.assertEqual(y1, 2)
        self.assertEqual(x2, x1)
        self.assertEqual(y2, y1 + 3)

    def test_ctor_in_python(self):
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

        @njit
        def idenity(obj):
            return obj

        @njit
        def retrieve_attributes(obj):
            return obj.x, obj.y, obj.arr

        arr = np.arange(10, dtype=np.float32)
        obj = Float2AndArray(1, 2, arr)
        self.assertEqual(obj._meminfo.refcount, 1)
        self.assertEqual(obj._meminfo.data, obj._dataptr)
        self.assertEqual(obj._typ.class_type, Float2AndArray.class_type)

        # Use jit class instance in numba
        other = idenity(obj)
        self.assertEqual(obj._meminfo.refcount, 2)
        self.assertEqual(other._meminfo.refcount, 2)
        self.assertEqual(other._meminfo.data, other._dataptr)
        self.assertEqual(other._meminfo.data, obj._meminfo.data)

        # Check dtor
        del other
        self.assertEqual(obj._meminfo.refcount, 1)

        # Check attributes
        out_x, out_y, out_arr = retrieve_attributes(obj)
        self.assertEqual(out_x, 1)
        self.assertEqual(out_y, 2)
        self.assertIs(out_arr, arr)




if __name__ == '__main__':
    unittest.main()
