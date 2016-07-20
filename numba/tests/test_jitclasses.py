from __future__ import absolute_import, print_function, division

from collections import OrderedDict
import ctypes
import sys

import numpy as np

from numba import types
from numba import (float32, float64, int16, int32, boolean, deferred_type,
                   optional)
from numba import njit, typeof, errors
from numba import unittest_support as unittest
from numba import jitclass
from .support import TestCase, MemoryLeakMixin, tag
from numba.jitclass import _box
from numba.runtime.nrt import MemInfo
from numba.six import assertRegex


def _get_meminfo(box):
    ptr = _box.box_get_meminfoptr(box)
    mi = MemInfo(ptr)
    mi.acquire()
    return mi


class TestJitClass(TestCase, MemoryLeakMixin):

    def _check_spec(self, spec):
        @jitclass(spec)
        class Test(object):

            def __init__(self):
                pass

        clsty = Test.class_type.instance_type
        names = list(clsty.struct.keys())
        values = list(clsty.struct.values())
        self.assertEqual(names[0], 'x')
        self.assertEqual(names[1], 'y')
        self.assertEqual(values[0], int32)
        self.assertEqual(values[1], float32)

    def test_ordereddict_spec(self):
        spec = OrderedDict()
        spec['x'] = int32
        spec['y'] = float32
        self._check_spec(spec)

    def test_list_spec(self):
        spec = [('x', int32),
                ('y', float32)]
        self._check_spec(spec)

    def test_spec_errors(self):
        spec1 = [('x', int), ('y', float32[:])]
        spec2 = [(1, int32), ('y', float32[:])]

        class Test(object):

            def __init__(self):
                pass

        with self.assertRaises(TypeError) as raises:
            jitclass(spec1)(Test)
        self.assertIn("spec values should be Numba type instances",
                      str(raises.exception))
        with self.assertRaises(TypeError) as raises:
            jitclass(spec2)(Test)
        self.assertEqual(str(raises.exception),
                         "spec keys should be strings, got 1")

    def _make_Float2AndArray(self):
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

        return Float2AndArray

    def _make_Vector2(self):
        spec = OrderedDict()
        spec['x'] = int32
        spec['y'] = int32

        @jitclass(spec)
        class Vector2(object):

            def __init__(self, x, y):
                self.x = x
                self.y = y

        return Vector2

    @tag('important')
    def test_jit_class_1(self):
        Float2AndArray = self._make_Float2AndArray()
        Vector2 = self._make_Vector2()

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

    @tag('important')
    def test_jitclass_usage_from_python(self):
        Float2AndArray = self._make_Float2AndArray()

        @njit
        def identity(obj):
            return obj

        @njit
        def retrieve_attributes(obj):
            return obj.x, obj.y, obj.arr

        arr = np.arange(10, dtype=np.float32)
        obj = Float2AndArray(1, 2, arr)
        obj_meminfo = _get_meminfo(obj)
        self.assertEqual(obj_meminfo.refcount, 2)
        self.assertEqual(obj_meminfo.data, _box.box_get_dataptr(obj))
        self.assertEqual(obj._numba_type_.class_type,
                         Float2AndArray.class_type)
        # Use jit class instance in numba
        other = identity(obj)
        other_meminfo = _get_meminfo(other)  # duplicates MemInfo object to obj
        self.assertEqual(obj_meminfo.refcount, 4)
        self.assertEqual(other_meminfo.refcount, 4)
        self.assertEqual(other_meminfo.data, _box.box_get_dataptr(other))
        self.assertEqual(other_meminfo.data, obj_meminfo.data)

        # Check dtor
        del other, other_meminfo
        self.assertEqual(obj_meminfo.refcount, 2)

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

    @tag('important')
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

        first_meminfo = _get_meminfo(first)
        second_meminfo = _get_meminfo(second)
        self.assertEqual(first_meminfo.refcount, 3)
        self.assertEqual(second.next.data, first.data)
        self.assertEqual(first_meminfo.refcount, 3)
        self.assertEqual(second_meminfo.refcount, 2)

        # Test using deferred type as argument
        first_val = second.get_next_data()
        self.assertEqual(first_val, first.data)

        # Check ownership
        self.assertEqual(first_meminfo.refcount, 3)
        del second, second_meminfo
        self.assertEqual(first_meminfo.refcount, 2)

    def test_c_structure(self):
        spec = OrderedDict()
        spec['a'] = int32
        spec['b'] = int16
        spec['c'] = float64

        @jitclass(spec)
        class Struct(object):

            def __init__(self, a, b, c):
                self.a = a
                self.b = b
                self.c = c

        st = Struct(0xabcd, 0xef, 3.1415)

        class CStruct(ctypes.Structure):
            _fields_ = [
                ('a', ctypes.c_int32),
                ('b', ctypes.c_int16),
                ('c', ctypes.c_double),
            ]

        ptr = ctypes.c_void_p(_box.box_get_dataptr(st))
        cstruct = ctypes.cast(ptr, ctypes.POINTER(CStruct))[0]
        self.assertEqual(cstruct.a, st.a)
        self.assertEqual(cstruct.b, st.b)
        self.assertEqual(cstruct.c, st.c)

    def test_is(self):
        Vector = self._make_Vector2()
        vec_a = Vector(1, 2)
        vec_b = Vector(1, 2)

        @njit
        def do_is(a, b):
            return a is b

        with self.assertRaises(errors.LoweringError) as raises:
            # trigger compilation
            do_is(vec_a, vec_a)
        self.assertIn('no default `is` implementation', str(raises.exception))

    def test_isinstance(self):
        Vector2 = self._make_Vector2()
        vec = Vector2(1, 2)
        self.assertIsInstance(vec, Vector2)

    def test_subclassing(self):
        Vector2 = self._make_Vector2()
        with self.assertRaises(TypeError) as raises:
            class SubV(Vector2):
                pass
        self.assertEqual(str(raises.exception),
                         "cannot subclass from a jitclass")

    def test_base_class(self):
        class Base(object):

            def what(self):
                return self.attr

        @jitclass([('attr', int32)])
        class Test(Base):

            def __init__(self, attr):
                self.attr = attr

        obj = Test(123)
        self.assertEqual(obj.what(), 123)

    def test_globals(self):

        class Mine(object):
            constant = 123

            def __init__(self):
                pass

        with self.assertRaises(TypeError) as raises:
            jitclass(())(Mine)

        self.assertEqual(str(raises.exception),
                         "class members are not yet supported: constant")

    @tag('important')
    def test_user_getter_setter(self):
        @jitclass([('attr', int32)])
        class Foo(object):

            def __init__(self, attr):
                self.attr = attr

            @property
            def value(self):
                return self.attr + 1

            @value.setter
            def value(self, val):
                self.attr = val - 1

        foo = Foo(123)
        self.assertEqual(foo.attr, 123)
        # Getter
        self.assertEqual(foo.value, 123 + 1)
        # Setter
        foo.value = 789
        self.assertEqual(foo.attr, 789 - 1)
        self.assertEqual(foo.value, 789)

        # Test nopython mode usage of getter and setter
        @njit
        def bar(foo, val):
            a = foo.value
            foo.value = val
            b = foo.value
            c = foo.attr
            return a, b, c

        a, b, c = bar(foo, 567)
        self.assertEqual(a, 789)
        self.assertEqual(b, 567)
        self.assertEqual(c, 567 - 1)

    def test_user_deleter_error(self):
        class Foo(object):

            def __init__(self):
                pass

            @property
            def value(self):
                return 1

            @value.deleter
            def value(self):
                pass

        with self.assertRaises(TypeError) as raises:
            jitclass([])(Foo)
        self.assertEqual(str(raises.exception),
                         "deleter is not supported: value")

    def test_name_shadowing_error(self):
        class Foo(object):

            def __init__(self):
                pass

            @property
            def my_property(self):
                pass

            def my_method(self):
                pass

        with self.assertRaises(NameError) as raises:
            jitclass([('my_property', int32)])(Foo)
        self.assertEqual(str(raises.exception), 'name shadowing: my_property')

        with self.assertRaises(NameError) as raises:
            jitclass([('my_method', int32)])(Foo)
        self.assertEqual(str(raises.exception), 'name shadowing: my_method')

    def test_distinct_classes(self):
        # Different classes with the same names shouldn't confuse the compiler
        @jitclass([('x', int32)])
        class Foo(object):

            def __init__(self, x):
                self.x = x + 2

            def run(self):
                return self.x + 1

        FirstFoo = Foo

        @jitclass([('x', int32)])
        class Foo(object):

            def __init__(self, x):
                self.x = x - 2

            def run(self):
                return self.x - 1

        SecondFoo = Foo
        foo = FirstFoo(5)
        self.assertEqual(foo.x, 7)
        self.assertEqual(foo.run(), 8)
        foo = SecondFoo(5)
        self.assertEqual(foo.x, 3)
        self.assertEqual(foo.run(), 2)

    def test_parameterized(self):
        class MyClass(object):

            def __init__(self, value):
                self.value = value

        def create_my_class(value):
            cls = jitclass([('value', typeof(value))])(MyClass)
            return cls(value)

        a = create_my_class(123)
        self.assertEqual(a.value, 123)

        b = create_my_class(12.3)
        self.assertEqual(b.value, 12.3)

        c = create_my_class(np.array([123]))
        np.testing.assert_equal(c.value, [123])

        d = create_my_class(np.array([12.3]))
        np.testing.assert_equal(d.value, [12.3])

    @tag('important')
    def test_protected_attrs(self):
        spec = {
            'value': int32,
            '_value': float32,
            '__value': int32,
            '__value__': int32,
        }

        @jitclass(spec)
        class MyClass(object):

            def __init__(self, value):
                self.value = value
                self._value = value / 2
                self.__value = value * 2
                self.__value__ = value - 1

            @property
            def private_value(self):
                return self.__value

            @property
            def _inner_value(self):
                return self._value

            @_inner_value.setter
            def _inner_value(self, v):
                self._value = v

            @property
            def __private_value(self):
                return self.__value

            @__private_value.setter
            def __private_value(self, v):
                self.__value = v

            def swap_private_value(self, new):
                old = self.__private_value
                self.__private_value = new
                return old

            def _protected_method(self, factor):
                return self._value * factor

            def __private_method(self, factor):
                return self.__value * factor

            def check_private_method(self, factor):
                return self.__private_method(factor)


        value = 123
        inst = MyClass(value)
        # test attributes
        self.assertEqual(inst.value, value)
        self.assertEqual(inst._value, value / 2)
        self.assertEqual(inst.private_value, value * 2)
        # test properties
        self.assertEqual(inst._inner_value, inst._value)
        freeze_inst_value = inst._value
        inst._inner_value -= 1
        self.assertEqual(inst._inner_value, freeze_inst_value - 1)

        self.assertEqual(inst.swap_private_value(321), value * 2)
        self.assertEqual(inst.swap_private_value(value * 2), 321)
        # test methods
        self.assertEqual(inst._protected_method(3), inst._value * 3)
        self.assertEqual(inst.check_private_method(3), inst.private_value * 3)
        # test special
        self.assertEqual(inst.__value__, value - 1)
        inst.__value__ -= 100
        self.assertEqual(inst.__value__, value - 101)

        # test errors
        @njit
        def access_dunder(inst):
            return inst.__value

        with self.assertRaises(errors.UntypedAttributeError) as raises:
            access_dunder(inst)
        # It will appear as "_TestJitClass__value" because the `access_dunder`
        # is under the scope of 'TestJitClass'.
        self.assertIn('_TestJitClass__value', str(raises.exception))

        with self.assertRaises(AttributeError) as raises:
            access_dunder.py_func(inst)
        self.assertIn('_TestJitClass__value', str(raises.exception))

    @unittest.skipIf(sys.version_info < (3,), "Python 3-specific test")
    def test_annotations(self):
        """
        Methods with annotations should compile fine (issue #1911).
        """
        from .annotation_usecases import AnnotatedClass

        spec = {'x': int32}
        cls = jitclass(spec)(AnnotatedClass)

        obj = cls(5)
        self.assertEqual(obj.x, 5)
        self.assertEqual(obj.add(2), 7)


class TestJitClassSpecialMethods(TestCase, MemoryLeakMixin):

    @tag('important')
    def test_hash(self):
        spec = [('_value', int32)]

        @jitclass(spec)
        class MyClass(object):

            def __init__(self, value):
                self._value = value

            def __hash__(self):
                return self._value

            def call_hash(self):
                return hash(self)

        instty = MyClass.class_type.instance_type
        self.assertIsInstance(instty, types.Hashable)
        # Checks for bug in the instancecheck that makes this an Integer
        self.assertNotIsInstance(instty, types.Integer)

        inst = MyClass(value=123)
        self.assertEqual(hash(inst), inst._value)
        self.assertEqual(inst.call_hash(), hash(inst))

    def test_not_hash(self):
        spec = [('_value', int32)]

        @jitclass(spec)
        class MyClass(object):

            def __init__(self, value):
                self._value = value

            def call_hash(self):
                return hash(self)

        instty = MyClass.class_type.instance_type
        self.assertNotIsInstance(instty, types.Hashable)

        inst = MyClass(value=123)
        with self.assertRaises(errors.TypingError) as raises:
            inst.call_hash()
        errmsg = str(raises.exception)
        regex = r"Invalid usage of Function\(<built-in function hash>\)"
        assertRegex(self, errmsg, regex)

    @tag('important')
    def test_cmp_interface(self):
        @jitclass([])
        class Cmp(object):
            def __init__(self):
                pass

            def __eq__(self):
                pass

            def __ne__(self):
                pass

            def __lt__(self):
                pass

            def __gt__(self):
                pass

            def __le__(self):
                pass

            def __ge__(self):
                pass

        jctype = Cmp.class_type.instance_type
        self.assertTrue(isinstance(jctype, types.Eq))
        self.assertTrue(isinstance(jctype, types.Ordered))

        self.assertTrue(isinstance(jctype, types.UserEq))
        self.assertTrue(isinstance(jctype, types.UserOrdered))

        self.assertTrue(isinstance(jctype, types.ClassInstanceType))

        # Test minimal interface to be Eq and Ordered
        @jitclass([])
        class Cmp2(object):
            def __init__(self):
                pass

            def __eq__(self):
                pass

            def __lt__(self):
                pass

        jctype = Cmp2.class_type.instance_type
        self.assertTrue(isinstance(jctype, types.Eq))
        self.assertTrue(isinstance(jctype, types.Ordered))

        self.assertTrue(isinstance(jctype, types.UserEq))
        self.assertTrue(isinstance(jctype, types.UserOrdered))

        self.assertTrue(isinstance(jctype, types.ClassInstanceType))

    @tag('important')
    def test_not_cmp_interface(self):
        @jitclass([])
        class NotCmp(object):
            def __init__(self):
                pass

        jctype = NotCmp.class_type.instance_type
        self.assertFalse(isinstance(jctype, types.Eq))
        self.assertFalse(isinstance(jctype, types.Ordered))

        self.assertFalse(isinstance(jctype, types.UserEq))
        self.assertFalse(isinstance(jctype, types.UserOrdered))

        self.assertTrue(isinstance(jctype, types.ClassInstanceType))

    @tag('important')
    def test_eq(self):
        spec = [('_value', int32),
                ('use_count', int32)]

        @jitclass(spec)
        class MyClass(object):

            def __init__(self, value):
                self._value = value
                self.use_count = 0

            def __eq__(self, other):
                self.use_count += 1
                if isinstance(other, MyClass):
                    return self._value == other._value

        @jitclass(spec)
        class MyOtherClass(object):

            def __init__(self, value):
                self._value = value
                self.use_count = 0

            def __eq__(self, other):
                self.use_count += 1
                return self._value == other._value

        myclass_insttype = MyClass.class_type.instance_type
        myotherclass_insttype = MyOtherClass.class_type.instance_type

        self.assertTrue(isinstance(myclass_insttype, types.UserEq))
        self.assertTrue(isinstance(myclass_insttype, types.Eq))

        self.assertTrue(isinstance(myotherclass_insttype, types.UserEq))
        self.assertTrue(isinstance(myotherclass_insttype, types.Eq))

        ai = MyClass(value=123)
        bi = MyClass(value=123)
        ci = MyClass(value=321)
        di = MyOtherClass(value=ai._value)

        self.assertEqual(ai.use_count, 0)
        self.assertEqual(bi.use_count, 0)
        self.assertEqual(ci.use_count, 0)
        self.assertEqual(di.use_count, 0)

        self.assertEqual(ai, bi)
        self.assertEqual(ai.use_count, 1)

        # notice the asymmetric __eq__ due to the instancecheck in MyClass
        self.assertEqual(di, ai)
        self.assertEqual(di.use_count, 1)

        @njit
        def check_equality(a, b):
            return a == b

        self.assertTrue(check_equality(ai, bi))
        self.assertEqual(ai.use_count, 2)

        self.assertFalse(check_equality(ai, ci))
        self.assertEqual(ai.use_count, 3)

        # notice the asymmetric __eq__ due to the instancecheck in MyClass
        self.assertFalse(check_equality(ai, di))
        self.assertEqual(ai.use_count, 4)

        self.assertTrue(check_equality(di, ai))
        self.assertEqual(di.use_count, 2)

        @njit
        def check_inequality(a, b):
            # not_equal provided via default __eq__
            return a != b

        self.assertFalse(check_inequality(ai, bi))
        self.assertEqual(ai.use_count, 5)

        self.assertTrue(check_inequality(ai, ci))
        self.assertEqual(ai.use_count, 6)

        # notice the asymmetric __eq__ due to the instancecheck in MyClass
        self.assertTrue(check_inequality(ai, di))
        self.assertEqual(ai.use_count, 7)

        self.assertFalse(check_inequality(di, ai))
        self.assertEqual(di.use_count, 3)

        self.assertEqual(bi.use_count, 0)
        self.assertEqual(ci.use_count, 0)

    @tag('important')
    def test_ne(self):
        spec = [('_value', int32),
                ('use_count', int32)]

        @jitclass(spec)
        class MyClass(object):

            def __init__(self, value):
                self._value = value
                self.use_count = 0

            def __eq__(self, other):
                return not (self != other)

            def __ne__(self, other):
                self.use_count += 1
                if isinstance(other, MyClass):
                    return self._value != other._value
                return True

        @jitclass(spec)
        class MyOtherClass(object):

            def __init__(self, value):
                self._value = value
                self.use_count = 0

            def __eq__(self, other):
                return not (self != other)

            def __ne__(self, other):
                self.use_count += 1
                return self._value != other._value

        myclass_insttype = MyClass.class_type.instance_type
        myotherclass_insttype = MyOtherClass.class_type.instance_type

        self.assertTrue(isinstance(myclass_insttype, types.UserEq))
        self.assertTrue(isinstance(myclass_insttype, types.Eq))

        self.assertTrue(isinstance(myotherclass_insttype, types.UserEq))
        self.assertTrue(isinstance(myotherclass_insttype, types.Eq))

        ai = MyClass(value=123)
        bi = MyClass(value=123)
        ci = MyClass(value=321)
        di = MyOtherClass(value=ai._value)

        self.assertEqual(ai.use_count, 0)
        self.assertEqual(bi.use_count, 0)
        self.assertEqual(ci.use_count, 0)
        self.assertEqual(di.use_count, 0)

        # there're no default __eq__ that uses __ne__; it will fallback to `is`
        self.assertEqual(ai, bi)
        self.assertEqual(ai.use_count, 1)
        self.assertEqual(bi.use_count, 0)

        self.assertFalse(ai != bi)
        self.assertEqual(ai.use_count, 2)

        self.assertNotEqual(ai, ci)
        self.assertEqual(ai.use_count, 3)

        # notice the asymmetric __eq__ due to the instancecheck in MyClass
        self.assertNotEqual(ai, di)
        self.assertEqual(ai.use_count, 4)
        self.assertFalse(di != ai)
        self.assertEqual(di.use_count, 1)

        @njit
        def check_inequality(a, b):
            # not_equal provided via default __eq__
            return a != b

        self.assertFalse(check_inequality(ai, bi))
        self.assertEqual(ai.use_count, 5)

        self.assertTrue(check_inequality(ai, ci))
        self.assertEqual(ai.use_count, 6)

        # notice the asymmetric __eq__ due to the instancecheck in MyClass
        self.assertTrue(check_inequality(ai, di))
        self.assertEqual(ai.use_count, 7)

        self.assertFalse(check_inequality(di, ai))
        self.assertEqual(di.use_count, 2)

        @njit
        def check_equality(a, b):
            return a == b

        self.assertTrue(check_equality(ai, bi))
        self.assertEqual(ai.use_count, 8)

        self.assertFalse(check_equality(ai, di))
        self.assertEqual(ai.use_count, 9)

        class BadClass(object):
            def __init__(self):
                pass

            def __ne__(self, other):
                pass

        with self.assertRaises(TypeError) as raises:
            jitclass(spec)(BadClass)
        self.assertEqual(str(raises.exception),
                         "class defined `__ne__` but not `__eq__`")

    def test_reflected_eq(self):
        spec = [('_value', int32),
                ('use_count', int32)]

        @jitclass(spec)
        class Apple(object):

            def __init__(self, value):
                self._value = value
                self.use_count = 0

            def __eq__(self, other):
                self.use_count += 1
                return self._value == other._value

        @jitclass(spec)
        class Berry(object):

            def __init__(self, value):
                self._value = value

        ai = Apple(value=123)
        bi = Berry(value=123)

        self.assertTrue(isinstance(typeof(ai), types.Eq))
        self.assertTrue(isinstance(typeof(ai), types.UserEq))

        self.assertFalse(isinstance(typeof(bi), types.Eq))
        self.assertFalse(isinstance(typeof(bi), types.UserEq))

        # the values are equal
        self.assertEqual(ai._value, bi._value)
        self.assertEqual(ai.use_count, 0)
        # equality is provided in this direction
        self.assertEqual(ai, bi)
        self.assertEqual(ai.use_count, 1)
        # reflected equality
        self.assertEqual(bi, ai)
        self.assertEqual(ai.use_count, 2)

        # check reflected equality in jitted code
        @njit
        def check_equality(x, y):
            return x == y

        self.assertTrue(check_equality(ai, bi))
        self.assertEqual(ai.use_count, 3)

        self.assertTrue(check_equality(bi, ai))
        self.assertEqual(ai.use_count, 4)

    def test_reflected_ne(self):
        spec = [('_value', int32),
                ('use_count', int32)]

        @jitclass(spec)
        class Apple(object):

            def __init__(self, value):
                self._value = value
                self.use_count = 0

            def __eq__(self, other):
                return not (self != other)

            def __ne__(self, other):
                self.use_count += 1
                return self._value != other._value

        @jitclass(spec)
        class Berry(object):

            def __init__(self, value):
                self._value = value

        ai = Apple(value=123)
        bi = Berry(value=321)

        # the values are not equal
        self.assertNotEqual(ai._value, bi._value)
        self.assertEqual(ai.use_count, 0)
        # inequality is provided in this direction
        self.assertNotEqual(ai, bi)
        self.assertEqual(ai.use_count, 1)
        # reflected inequality
        self.assertNotEqual(bi, ai)
        self.assertEqual(ai.use_count, 2)

        # check reflected equality in jitted code
        @njit
        def check_inequality(x, y):
            return x != y

        self.assertTrue(check_inequality(ai, bi))
        self.assertEqual(ai.use_count, 3)

        self.assertTrue(check_inequality(bi, ai))
        self.assertEqual(ai.use_count, 4)

    def test_lt(self):
        spec = [('_value', int32),
                ('use_count', int32)]

        @jitclass(spec)
        class Apple(object):
            def __init__(self, value):
                self._value = value
                self.use_count = 0

            def __lt__(self, other):
                self.use_count += 1
                return self._value < other._value

        @jitclass(spec)
        class Berry(object):
            def __init__(self, value):
                self._value = value

        ai = Apple(123)
        bi = Berry(124)

        self.assertTrue(isinstance(typeof(ai), types.UserOrdered))
        self.assertTrue(isinstance(typeof(ai), types.Ordered))

        self.assertFalse(isinstance(typeof(bi), types.UserOrdered))
        self.assertFalse(isinstance(typeof(bi), types.Ordered))

        self.assertEqual(ai.use_count, 0)
        self.assertTrue(ai < bi)
        self.assertEqual(ai.use_count, 1)
        self.assertTrue(bi > ai)
        self.assertEqual(ai.use_count, 2)

        @njit
        def check_lessthan(x, y):
            return x < y

        self.assertTrue(check_lessthan(ai, bi))
        self.assertEqual(ai.use_count, 3)

        # test reflection
        # __gt__ is the reflection of __lt__
        @njit
        def check_greaterthan(x, y):
            return x > y

        self.assertTrue(check_greaterthan(bi, ai))
        self.assertEqual(ai.use_count, 4)


    def test_gt(self):
        spec = [('_value', int32),
                ('use_count', int32)]

        @jitclass(spec)
        class Apple(object):
            def __init__(self, value):
                self._value = value
                self.use_count = 0

            def __gt__(self, other):
                self.use_count += 1
                return self._value > other._value

        @jitclass(spec)
        class Berry(object):
            def __init__(self, value):
                self._value = value


        ai = Apple(123)
        bi = Berry(122)

        self.assertTrue(isinstance(typeof(ai), types.UserOrdered))
        self.assertTrue(isinstance(typeof(ai), types.Ordered))

        self.assertFalse(isinstance(typeof(bi), types.UserOrdered))
        self.assertFalse(isinstance(typeof(bi), types.Ordered))

        self.assertEqual(ai.use_count, 0)
        self.assertTrue(ai > bi)
        self.assertEqual(ai.use_count, 1)
        self.assertTrue(bi < ai)
        self.assertEqual(ai.use_count, 2)

        @njit
        def check_greaterthan(x, y):
            return x > y

        self.assertTrue(check_greaterthan(ai, bi))
        self.assertEqual(ai.use_count, 3)

        # test reflection
        # __lt__ is the reflection of __gt__
        @njit
        def check_lessthan(x, y):
            return x < y

        self.assertTrue(check_lessthan(bi, ai))
        self.assertEqual(ai.use_count, 4)


    def test_le(self):
        spec = [('_value', int32),
                ('use_count', int32)]

        @jitclass(spec)
        class Apple(object):
            def __init__(self, value):
                self._value = value
                self.use_count = 0

            def __le__(self, other):
                self.use_count += 1
                return self._value <= other._value

        @jitclass(spec)
        class Berry(object):
            def __init__(self, value):
                self._value = value


        ai = Apple(123)
        bi = Berry(124)

        self.assertTrue(isinstance(typeof(ai), types.UserOrdered))
        self.assertTrue(isinstance(typeof(ai), types.Ordered))

        self.assertFalse(isinstance(typeof(bi), types.UserOrdered))
        self.assertFalse(isinstance(typeof(bi), types.Ordered))

        self.assertEqual(ai.use_count, 0)
        self.assertTrue(ai <= bi)
        self.assertEqual(ai.use_count, 1)
        self.assertTrue(bi >= ai)
        self.assertEqual(ai.use_count, 2)

        @njit
        def check_lessequal(x, y):
            return x <= y

        self.assertTrue(check_lessequal(ai, bi))
        self.assertEqual(ai.use_count, 3)

        # test reflection
        # __ge__ is the reflection of __le__
        @njit
        def check_greaterequal(x, y):
            return x >= y

        self.assertTrue(check_greaterequal(bi, ai))
        self.assertEqual(ai.use_count, 4)

    def test_ge(self):
        spec = [('_value', int32),
                ('use_count', int32)]

        @jitclass(spec)
        class Apple(object):
            def __init__(self, value):
                self._value = value
                self.use_count = 0

            def __ge__(self, other):
                self.use_count += 1
                return self._value >= other._value

        @jitclass(spec)
        class Berry(object):
            def __init__(self, value):
                self._value = value


        ai = Apple(123)
        bi = Berry(122)

        self.assertTrue(isinstance(typeof(ai), types.UserOrdered))
        self.assertTrue(isinstance(typeof(ai), types.Ordered))

        self.assertFalse(isinstance(typeof(bi), types.UserOrdered))
        self.assertFalse(isinstance(typeof(bi), types.Ordered))

        self.assertEqual(ai.use_count, 0)
        self.assertTrue(ai >= bi)
        self.assertEqual(ai.use_count, 1)
        self.assertTrue(bi <= ai)
        self.assertEqual(ai.use_count, 2)

        @njit
        def check_greaterequal(x, y):
            return x >= y

        self.assertTrue(check_greaterequal(ai, bi))
        self.assertEqual(ai.use_count, 3)

        # test reflection
        # __le__ is the reflection of __ge__
        @njit
        def check_lessequal(x, y):
            return x <= y

        self.assertTrue(check_lessequal(bi, ai))
        self.assertEqual(ai.use_count, 4)

    def test_default_ordering(self):
        spec = [('_value', int32),
                ('use_count', int32)]

        @jitclass(spec)
        class Apple(object):
            def __init__(self, value):
                self._value = value
                self.use_count = 0

            def __lt__(self, other):
                self.use_count += 1
                return self._value < other._value

            def __eq__(self, other):
                self.use_count += 1
                return self._value == other._value

        ai = Apple(123)
        bi = Apple(122)

        self.assertTrue(isinstance(typeof(ai), types.UserOrdered))
        self.assertTrue(isinstance(typeof(ai), types.Ordered))
        self.assertTrue(isinstance(typeof(ai), types.UserEq))
        self.assertTrue(isinstance(typeof(ai), types.Eq))

        self.assertEqual(ai.use_count, 0)
        self.assertEqual(bi.use_count, 0)

        self.assertFalse(ai < bi)
        self.assertEqual(ai.use_count, 1)
        self.assertEqual(bi.use_count, 0)

        self.assertFalse(ai <= bi)
        self.assertEqual(ai.use_count, 3)  # increment by 2
        self.assertEqual(bi.use_count, 0)

        self.assertTrue(ai >= bi)
        self.assertEqual(ai.use_count, 4)  # increment by 1
        self.assertEqual(bi.use_count, 0)

        self.assertTrue(ai > bi)
        self.assertEqual(ai.use_count, 6)  # increment by 2
        self.assertEqual(bi.use_count, 0)

        self.assertFalse(ai == bi)
        self.assertEqual(ai.use_count, 7)  # increment by 1
        self.assertEqual(bi.use_count, 0)

        self.assertTrue(ai != bi)
        self.assertEqual(ai.use_count, 8)  # increment by 1
        self.assertEqual(bi.use_count, 0)


if __name__ == '__main__':
    unittest.main()
