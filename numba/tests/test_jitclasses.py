import numpy as np
from numba import float32, int32, boolean, deferred_type, optional
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

    def test_jitclass_usage_from_python(self):
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
                return val

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
        self.assertEqual(obj._numba_type_.class_type, Float2AndArray.class_type)
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

        # Access attributes from python
        self.assertEqual(obj.x, 1)
        self.assertEqual(obj.y, 2)
        self.assertIs(obj.arr, arr)

        # Access methods from python
        self.assertEqual(obj.add(123), 123)
        self.assertEqual(obj.x, 1 + 123)
        self.assertEqual(obj.y, 2 + 123)

        # Setter from python
        obj.x = 333
        obj.y = 444
        obj.arr = newarr = np.arange(5, dtype=np.float32)
        self.assertEqual(obj.x, 333)
        self.assertEqual(obj.y, 444)
        self.assertIs(obj.arr, newarr)

    def test_jitclass_datalayout(self):
        spec = OrderedDict()
        # Boolean has different layout as value vs data
        spec['val'] = boolean

        @jitclass(spec)
        class Foo(object):
            def __init__(self, val):
                self.val = val

        self.assertTrue(Foo(True).val)
        self.assertFalse(Foo(False).val)

    def test_deferred_type(self):
        node_type = deferred_type()

        spec = OrderedDict()
        spec['data'] = float32
        spec['next'] = optional(node_type)

        @njit
        def get_data(node):
            return node.data

        @jitclass(spec)
        class LinkedNode(object):
            def __init__(self, data, next):
                self.data = data
                self.next = next

            def get_next_data(self):
                # use deferred type as argument
                return get_data(self.next)

        node_type.define(LinkedNode.class_type.instance_type)

        first = LinkedNode(123, None)
        self.assertEqual(first.data, 123)
        self.assertIsNone(first.next)

        second = LinkedNode(321, first)
        self.assertEqual(first._meminfo.refcount, 2)
        self.assertEqual(second.next.data, first.data)
        self.assertEqual(first._meminfo.refcount, 2)
        self.assertEqual(second._meminfo.refcount, 1)

        # Test using deferred type as argument
        first_val = second.get_next_data()
        self.assertEqual(first_val, first.data)

        # Check ownership
        self.assertEqual(first._meminfo.refcount, 2)
        del second
        self.assertEqual(first._meminfo.refcount, 1)


if __name__ == '__main__':
    unittest.main()
