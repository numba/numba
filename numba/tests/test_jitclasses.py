import ctypes
import pickle
import random
import typing as pt
import unittest

from collections import OrderedDict

import numpy as np
from numba import (boolean, deferred_type, float32, float64, int16, int32,
                   njit, optional, typeof)
from numba.core import errors, types
from numba.core.dispatcher import Dispatcher
from numba.core.errors import LoweringError
from numba.core.runtime.nrt import MemInfo
from numba.experimental import jitclass
from numba.experimental.jitclass import _box
from numba.experimental.jitclass.base import JitClassType
from numba.tests.support import MemoryLeakMixin, TestCase


class TestClass1(object):
    def __init__(self, x, y, z=1, *, a=5):
        self.x = x
        self.y = y
        self.z = z
        self.a = a


class TestClass2(object):
    def __init__(self, x, y, z=1, *args, a=5):
        self.x = x
        self.y = y
        self.z = z
        self.args = args
        self.a = a


def _get_meminfo(box):
    ptr = _box.box_get_meminfoptr(box)
    mi = MemInfo(ptr)
    mi.acquire()
    return mi


class TestJitClass(TestCase, MemoryLeakMixin):

    def _check_spec(self, spec=None, test_cls=None, all_expected=None):
        if test_cls is None:
            @jitclass(spec)
            class Test(object):

                def __init__(self):
                    pass
            test_cls = Test

        clsty = test_cls.class_type.instance_type
        names = list(clsty.struct.keys())
        values = list(clsty.struct.values())

        if all_expected is None:
            if isinstance(spec, OrderedDict):
                all_expected = spec.items()
            else:
                all_expected = spec

        assert all_expected is not None

        self.assertEqual(len(names), len(all_expected))
        for got, expected in zip(zip(names, values), all_expected):
            self.assertEqual(got[0], expected[0])
            self.assertEqual(got[1], expected[1])

    def test_ordereddict_spec(self):
        spec = OrderedDict()
        spec["x"] = int32
        spec["y"] = float32
        self._check_spec(spec)

    def test_list_spec(self):
        spec = [("x", int32),
                ("y", float32)]
        self._check_spec(spec)

    def test_type_annotations(self):
        spec = [("x", int32)]

        @jitclass(spec)
        class Test1(object):
            x: int
            y: pt.List[float]

            def __init__(self):
                pass

        self._check_spec(spec, Test1, spec + [("y", types.ListType(float64))])

    def test_type_annotation_inheritance(self):

        class Foo:
            x: int

        @jitclass
        class Bar(Foo):
            y: float

            def __init__(self, value: float) -> None:
                self.x = int(value)
                self.y = value

        self._check_spec(
            test_cls=Bar, all_expected=[("x", typeof(0)), ("y", typeof(0.0))]
        )

    def test_spec_errors(self):
        spec1 = [("x", int), ("y", float32[:])]
        spec2 = [(1, int32), ("y", float32[:])]

        class Test(object):

            def __init__(self):
                pass

        with self.assertRaises(TypeError) as raises:
            jitclass(Test, spec1)
        self.assertIn("spec values should be Numba type instances",
                      str(raises.exception))
        with self.assertRaises(TypeError) as raises:
            jitclass(Test, spec2)
        self.assertEqual(str(raises.exception),
                         "spec keys should be strings, got 1")

    def test_init_errors(self):

        @jitclass([])
        class Test:
            def __init__(self):
                return 7

        with self.assertRaises(errors.TypingError) as raises:
            Test()

        self.assertIn("__init__() should return None, not",
                      str(raises.exception))

    def _make_Float2AndArray(self):
        spec = OrderedDict()
        spec["x"] = float32
        spec["y"] = float32
        spec["arr"] = float32[:]

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
        spec["x"] = int32
        spec["y"] = int32

        @jitclass(spec)
        class Vector2(object):

            def __init__(self, x, y):
                self.x = x
                self.y = y

        return Vector2

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
        spec["val"] = boolean

        @jitclass(spec)
        class Foo(object):

            def __init__(self, val):
                self.val = val

        self.assertTrue(Foo(True).val)
        self.assertFalse(Foo(False).val)

    def test_deferred_type(self):
        node_type = deferred_type()

        spec = OrderedDict()
        spec["data"] = float32
        spec["next"] = optional(node_type)

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

            def append_to_tail(self, other):
                cur = self
                while cur.next is not None:
                    cur = cur.next
                cur.next = other

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

        # Check setattr (issue #2606)
        self.assertIsNone(first.next)
        second.append_to_tail(LinkedNode(567, None))
        self.assertIsNotNone(first.next)
        self.assertEqual(first.next.data, 567)
        self.assertIsNone(first.next.next)
        second.append_to_tail(LinkedNode(678, None))
        self.assertIsNotNone(first.next.next)
        self.assertEqual(first.next.next.data, 678)

        # Check ownership
        self.assertEqual(first_meminfo.refcount, 3)
        del second, second_meminfo
        self.assertEqual(first_meminfo.refcount, 2)

    def test_c_structure(self):
        spec = OrderedDict()
        spec["a"] = int32
        spec["b"] = int16
        spec["c"] = float64

        @jitclass(spec)
        class Struct(object):

            def __init__(self, a, b, c):
                self.a = a
                self.b = b
                self.c = c

        st = Struct(0xabcd, 0xef, 3.1415)

        class CStruct(ctypes.Structure):
            _fields_ = [
                ("a", ctypes.c_int32),
                ("b", ctypes.c_int16),
                ("c", ctypes.c_double),
            ]

        ptr = ctypes.c_void_p(_box.box_get_dataptr(st))
        cstruct = ctypes.cast(ptr, ctypes.POINTER(CStruct))[0]
        self.assertEqual(cstruct.a, st.a)
        self.assertEqual(cstruct.b, st.b)
        self.assertEqual(cstruct.c, st.c)

    def test_is(self):
        Vector = self._make_Vector2()
        vec_a = Vector(1, 2)

        @njit
        def do_is(a, b):
            return a is b

        with self.assertRaises(LoweringError) as raises:
            # trigger compilation
            do_is(vec_a, vec_a)
        self.assertIn("no default `is` implementation", str(raises.exception))

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

        @jitclass([("attr", int32)])
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
            jitclass(Mine)

        self.assertEqual(str(raises.exception),
                         "class members are not yet supported: constant")

    def test_user_getter_setter(self):
        @jitclass([("attr", int32)])
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
            jitclass(Foo)
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
            jitclass(Foo, [("my_property", int32)])
        self.assertEqual(str(raises.exception), "name shadowing: my_property")

        with self.assertRaises(NameError) as raises:
            jitclass(Foo, [("my_method", int32)])
        self.assertEqual(str(raises.exception), "name shadowing: my_method")

    def test_distinct_classes(self):
        # Different classes with the same names shouldn't confuse the compiler
        @jitclass([("x", int32)])
        class Foo(object):

            def __init__(self, x):
                self.x = x + 2

            def run(self):
                return self.x + 1

        FirstFoo = Foo

        @jitclass([("x", int32)])
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
            cls = jitclass(MyClass, [("value", typeof(value))])
            return cls(value)

        a = create_my_class(123)
        self.assertEqual(a.value, 123)

        b = create_my_class(12.3)
        self.assertEqual(b.value, 12.3)

        c = create_my_class(np.array([123]))
        np.testing.assert_equal(c.value, [123])

        d = create_my_class(np.array([12.3]))
        np.testing.assert_equal(d.value, [12.3])

    def test_protected_attrs(self):
        spec = {
            "value": int32,
            "_value": float32,
            "__value": int32,
            "__value__": int32,
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

        with self.assertRaises(errors.TypingError) as raises:
            access_dunder(inst)
        # It will appear as "_TestJitClass__value" because the `access_dunder`
        # is under the scope of "TestJitClass".
        self.assertIn("_TestJitClass__value", str(raises.exception))

        with self.assertRaises(AttributeError) as raises:
            access_dunder.py_func(inst)
        self.assertIn("_TestJitClass__value", str(raises.exception))

    def test_annotations(self):
        """
        Methods with annotations should compile fine (issue #1911).
        """
        from .annotation_usecases import AnnotatedClass

        spec = {"x": int32}
        cls = jitclass(AnnotatedClass, spec)

        obj = cls(5)
        self.assertEqual(obj.x, 5)
        self.assertEqual(obj.add(2), 7)

    def test_docstring(self):

        @jitclass
        class Apple(object):
            "Class docstring"

            def __init__(self):
                "init docstring"

            def foo(self):
                "foo method docstring"

            @property
            def aval(self):
                "aval property docstring"

        self.assertEqual(Apple.__doc__, "Class docstring")
        self.assertEqual(Apple.__init__.__doc__, "init docstring")
        self.assertEqual(Apple.foo.__doc__, "foo method docstring")
        self.assertEqual(Apple.aval.__doc__, "aval property docstring")

    def test_kwargs(self):
        spec = [("a", int32),
                ("b", float64)]

        @jitclass(spec)
        class TestClass(object):
            def __init__(self, x, y, z):
                self.a = x * y
                self.b = z

        x = 2
        y = 2
        z = 1.1
        kwargs = {"y": y, "z": z}
        tc = TestClass(x=2, **kwargs)
        self.assertEqual(tc.a, x * y)
        self.assertEqual(tc.b, z)

    def test_default_args(self):
        spec = [("x", int32),
                ("y", int32),
                ("z", int32)]

        @jitclass(spec)
        class TestClass(object):
            def __init__(self, x, y, z=1):
                self.x = x
                self.y = y
                self.z = z

        tc = TestClass(1, 2, 3)
        self.assertEqual(tc.x, 1)
        self.assertEqual(tc.y, 2)
        self.assertEqual(tc.z, 3)

        tc = TestClass(1, 2)
        self.assertEqual(tc.x, 1)
        self.assertEqual(tc.y, 2)
        self.assertEqual(tc.z, 1)

        tc = TestClass(y=2, z=5, x=1)
        self.assertEqual(tc.x, 1)
        self.assertEqual(tc.y, 2)
        self.assertEqual(tc.z, 5)

    def test_default_args_keyonly(self):
        spec = [("x", int32),
                ("y", int32),
                ("z", int32),
                ("a", int32)]

        TestClass = jitclass(TestClass1, spec)

        tc = TestClass(2, 3)
        self.assertEqual(tc.x, 2)
        self.assertEqual(tc.y, 3)
        self.assertEqual(tc.z, 1)
        self.assertEqual(tc.a, 5)

        tc = TestClass(y=4, x=2, a=42, z=100)
        self.assertEqual(tc.x, 2)
        self.assertEqual(tc.y, 4)
        self.assertEqual(tc.z, 100)
        self.assertEqual(tc.a, 42)

        tc = TestClass(y=4, x=2, a=42)
        self.assertEqual(tc.x, 2)
        self.assertEqual(tc.y, 4)
        self.assertEqual(tc.z, 1)
        self.assertEqual(tc.a, 42)

        tc = TestClass(y=4, x=2)
        self.assertEqual(tc.x, 2)
        self.assertEqual(tc.y, 4)
        self.assertEqual(tc.z, 1)
        self.assertEqual(tc.a, 5)

    def test_default_args_starargs_and_keyonly(self):
        spec = [("x", int32),
                ("y", int32),
                ("z", int32),
                ("args", types.UniTuple(int32, 2)),
                ("a", int32)]

        with self.assertRaises(errors.UnsupportedError) as raises:
            jitclass(TestClass2, spec)

        msg = "VAR_POSITIONAL argument type unsupported"
        self.assertIn(msg, str(raises.exception))

    def test_generator_method(self):
        spec = []

        @jitclass(spec)
        class TestClass(object):
            def __init__(self):
                pass

            def gen(self, niter):
                for i in range(niter):
                    yield np.arange(i)

        def expected_gen(niter):
            for i in range(niter):
                yield np.arange(i)

        for niter in range(10):
            for expect, got in zip(expected_gen(niter), TestClass().gen(niter)):
                self.assertPreciseEqual(expect, got)

    def test_getitem(self):
        spec = [("data", int32[:])]

        @jitclass(spec)
        class TestClass(object):
            def __init__(self):
                self.data = np.zeros(10, dtype=np.int32)

            def __setitem__(self, key, data):
                self.data[key] = data

            def __getitem__(self, key):
                return self.data[key]

        @njit
        def create_and_set_indices():
            t = TestClass()
            t[1] = 1
            t[2] = 2
            t[3] = 3
            return t

        @njit
        def get_index(t, n):
            return t[n]

        t = create_and_set_indices()
        self.assertEqual(get_index(t, 1), 1)
        self.assertEqual(get_index(t, 2), 2)
        self.assertEqual(get_index(t, 3), 3)

    def test_getitem_unbox(self):
        spec = [("data", int32[:])]

        @jitclass(spec)
        class TestClass(object):
            def __init__(self):
                self.data = np.zeros(10, dtype=np.int32)

            def __setitem__(self, key, data):
                self.data[key] = data

            def __getitem__(self, key):
                return self.data[key]

        t = TestClass()
        t[1] = 10

        @njit
        def set2return1(t):
            t[2] = 20
            return t[1]

        t_1 = set2return1(t)
        self.assertEqual(t_1, 10)
        self.assertEqual(t[2], 20)

    def test_getitem_complex_key(self):
        spec = [("data", int32[:, :])]

        @jitclass(spec)
        class TestClass(object):
            def __init__(self):
                self.data = np.zeros((10, 10), dtype=np.int32)

            def __setitem__(self, key, data):
                self.data[int(key.real), int(key.imag)] = data

            def __getitem__(self, key):
                return self.data[int(key.real), int(key.imag)]

        t = TestClass()

        t[complex(1, 1)] = 3

        @njit
        def get_key(t, real, imag):
            return t[complex(real, imag)]

        @njit
        def set_key(t, real, imag, data):
            t[complex(real, imag)] = data

        self.assertEqual(get_key(t, 1, 1), 3)
        set_key(t, 2, 2, 4)
        self.assertEqual(t[complex(2, 2)], 4)

    def test_getitem_tuple_key(self):
        spec = [("data", int32[:, :])]

        @jitclass(spec)
        class TestClass(object):
            def __init__(self):
                self.data = np.zeros((10, 10), dtype=np.int32)

            def __setitem__(self, key, data):
                self.data[key[0], key[1]] = data

            def __getitem__(self, key):
                return self.data[key[0], key[1]]

        t = TestClass()
        t[1, 1] = 11

        @njit
        def get11(t):
            return t[1, 1]

        @njit
        def set22(t, data):
            t[2, 2] = data

        self.assertEqual(get11(t), 11)
        set22(t, 22)
        self.assertEqual(t[2, 2], 22)

    def test_getitem_slice_key(self):
        spec = [("data", int32[:])]

        @jitclass(spec)
        class TestClass(object):
            def __init__(self):
                self.data = np.zeros(10, dtype=np.int32)

            def __setitem__(self, slc, data):
                self.data[slc.start] = data
                self.data[slc.stop] = data + slc.step

            def __getitem__(self, slc):
                return self.data[slc.start]

        t = TestClass()
        # set t.data[1] = 1 and t.data[5] = 2
        t[1:5:1] = 1

        self.assertEqual(t[1:1:1], 1)
        self.assertEqual(t[5:5:5], 2)

        @njit
        def get5(t):
            return t[5:6:1]

        self.assertEqual(get5(t), 2)

        # sets t.data[2] = data, and t.data[6] = data + 1
        @njit
        def set26(t, data):
            t[2:6:1] = data

        set26(t, 2)
        self.assertEqual(t[2:2:1], 2)
        self.assertEqual(t[6:6:1], 3)

    def test_jitclass_longlabel_not_truncated(self):
        # See issue #3872, llvm 7 introduced a max label length of 1024 chars
        # Numba ships patched llvm 7.1 (ppc64le) and patched llvm 8 to undo this
        # change, this test is here to make sure long labels are ok:
        alphabet = [chr(ord("a") + x) for x in range(26)]

        spec = [(letter * 10, float64) for letter in alphabet]
        spec.extend([(letter.upper() * 10, float64) for letter in alphabet])

        @jitclass(spec)
        class TruncatedLabel(object):
            def __init__(self,):
                self.aaaaaaaaaa = 10.

            def meth1(self):
                self.bbbbbbbbbb = random.gauss(self.aaaaaaaaaa, self.aaaaaaaaaa)

            def meth2(self):
                self.meth1()

        # unpatched LLVMs will raise here...
        TruncatedLabel().meth2()

    def test_pickling(self):
        @jitclass
        class PickleTestSubject(object):
            def __init__(self):
                pass

        inst = PickleTestSubject()
        ty = typeof(inst)
        self.assertIsInstance(ty, types.ClassInstanceType)
        pickled = pickle.dumps(ty)
        self.assertIs(pickle.loads(pickled), ty)

    def test_static_methods(self):
        @jitclass([("x", int32)])
        class Test1:
            def __init__(self, x):
                self.x = x

            def increase(self, y):
                self.x = self.add(self.x, y)
                return self.x

            @staticmethod
            def add(a, b):
                return a + b

            @staticmethod
            def sub(a, b):
                return a - b

        @jitclass([("x", int32)])
        class Test2:
            def __init__(self, x):
                self.x = x

            def increase(self, y):
                self.x = self.add(self.x, y)
                return self.x

            @staticmethod
            def add(a, b):
                return a - b

        self.assertIsInstance(Test1.add, Dispatcher)
        self.assertIsInstance(Test1.sub, Dispatcher)
        self.assertIsInstance(Test2.add, Dispatcher)
        self.assertNotEqual(Test1.add, Test2.add)

        self.assertEqual(3, Test1.add(1, 2))
        self.assertEqual(-1, Test2.add(1, 2))
        self.assertEqual(4, Test1.sub(6, 2))

        t1 = Test1(0)
        t2 = Test2(0)
        self.assertEqual(1, t1.increase(1))
        self.assertEqual(-1, t2.increase(1))
        self.assertEqual(2, t1.add(1, 1))
        self.assertEqual(0, t1.sub(1, 1))
        self.assertEqual(0, t2.add(1, 1))
        self.assertEqual(2j, t1.add(1j, 1j))
        self.assertEqual(1j, t1.sub(2j, 1j))
        self.assertEqual("foobar", t1.add("foo", "bar"))

        with self.assertRaises(AttributeError) as raises:
            Test2.sub(3, 1)
        self.assertIn("has no attribute 'sub'",
                      str(raises.exception))

        with self.assertRaises(TypeError) as raises:
            Test1.add(3)
        self.assertIn("not enough arguments: expected 2, got 1",
                      str(raises.exception))

        # Check error message for calling a static method as a class attr from
        # another method (currently unsupported).

        @jitclass([])
        class Test3:
            def __init__(self):
                pass

            @staticmethod
            def a_static_method(a, b):
                pass

            def call_static(self):
                return Test3.a_static_method(1, 2)

        invalid = Test3()
        with self.assertRaises(errors.TypingError) as raises:
            invalid.call_static()

        self.assertIn("Unknown attribute 'a_static_method'",
                      str(raises.exception))

    def test_jitclass_decorator_usecases(self):
        spec = OrderedDict(x=float64)

        @jitclass()
        class Test1:
            x: float

            def __init__(self):
                self.x = 0

        self.assertIsInstance(Test1, JitClassType)
        self.assertDictEqual(Test1.class_type.struct, spec)

        @jitclass(spec=spec)
        class Test2:

            def __init__(self):
                self.x = 0

        self.assertIsInstance(Test2, JitClassType)
        self.assertDictEqual(Test2.class_type.struct, spec)

        @jitclass
        class Test3:
            x: float

            def __init__(self):
                self.x = 0

        self.assertIsInstance(Test3, JitClassType)
        self.assertDictEqual(Test3.class_type.struct, spec)

        @jitclass(spec)
        class Test4:

            def __init__(self):
                self.x = 0

        self.assertIsInstance(Test4, JitClassType)
        self.assertDictEqual(Test4.class_type.struct, spec)

    def test_jitclass_function_usecases(self):
        spec = OrderedDict(x=float64)

        class AnnotatedTest:
            x: float

            def __init__(self):
                self.x = 0

        JitTest1 = jitclass(AnnotatedTest)
        self.assertIsInstance(JitTest1, JitClassType)
        self.assertDictEqual(JitTest1.class_type.struct, spec)

        class UnannotatedTest:

            def __init__(self):
                self.x = 0

        JitTest2 = jitclass(UnannotatedTest, spec)
        self.assertIsInstance(JitTest2, JitClassType)
        self.assertDictEqual(JitTest2.class_type.struct, spec)


if __name__ == "__main__":
    unittest.main()
