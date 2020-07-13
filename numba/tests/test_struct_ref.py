"""
Test mutable struct, aka, structref
"""
import numpy as np

from numba.core import types
from numba import njit
from numba.core import structref
from numba.extending import overload_method
from numba.tests.support import MemoryLeakMixin, TestCase



@structref.register
class MySimplerStructType(types.StructRef):
    pass


my_struct_ty = MySimplerStructType(
    fields=[("values", types.intp[:]), ("counter", types.intp)]
)

structref.define_boxing(MySimplerStructType, structref.StructRefProxy)


class MyStruct(structref.StructRefProxy):
    def __new__(cls, values, counter):
        # Define this method to customize the constructor.
        # The default takes `*args`. Customizing allow the use of keyword-arg.
        # The impl of the method calls `StructRefProxy.__new__`
        return structref.StructRefProxy.__new__(cls, values, counter)

    # The below defines wrappers for attributes and methods manually

    @property
    def values(self):
        return get_values(self)

    @property
    def counter(self):
        return get_counter(self)

    def testme(self, arg):
        return self.values * arg + self.counter


@structref.register
class MyStructType(types.StructRef):
    pass


# Call to define_proxy is needed to register the use of `MyStruct` as a
# PyObject proxy for creating a Numba-allocated structref.
# The `MyStruct` class and then be used in both jit-code and interpreted-code.
structref.define_proxy(
    MyStruct,
    MyStructType,
    ['values', 'counter'],
)


@njit
def my_struct(values, counter):
    st = structref.new(my_struct_ty)
    my_struct_init(st, values, counter)
    return st


@njit
def my_struct_init(self, values, counter):
    self.values = values
    self.counter = counter


@njit
def ctor_by_intrinsic(vs, ctr):
    st = my_struct(vs, counter=ctr)
    st.values += st.values
    st.counter *= ctr
    return st


@njit
def ctor_by_class(vs, ctr):
    return MyStruct(values=vs, counter=ctr)


@njit
def get_values(st):
    return st.values


@njit
def get_counter(st):
    return st.counter


@njit
def compute_fields(st):
    return st.values + st.counter


class TestStructRefBasic(MemoryLeakMixin, TestCase):
    def test_MySimplerStructType(self):
        vs = np.arange(10, dtype=np.intp)
        ctr = 13

        first_expected = vs + vs
        first_got = ctor_by_intrinsic(vs, ctr)
        # the returned instance is a structref.StructRefProxy
        # but not a MyStruct
        self.assertNotIsInstance(first_got, MyStruct)
        self.assertPreciseEqual(first_expected, get_values(first_got))

        second_expected = first_expected + (ctr * ctr)
        second_got = compute_fields(first_got)
        self.assertPreciseEqual(second_expected, second_got)

    def test_MyStructType(self):
        vs = np.arange(10, dtype=np.float64)
        ctr = 11

        first_expected = vs.copy()
        first_got = ctor_by_class(vs, ctr)
        self.assertIsInstance(first_got, MyStruct)
        self.assertPreciseEqual(first_expected, first_got.values)

        second_expected = first_expected + ctr
        second_got = compute_fields(first_got)
        self.assertPreciseEqual(second_expected, second_got)

        self.assertEqual(first_got.counter, ctr)


class TestStructRefExtending(MemoryLeakMixin, TestCase):
    def test_overload_method(self):
        @overload_method(MyStructType, "testme")
        def _(self, arg):
            def impl(self, arg):
                return self.values * arg + self.counter
            return impl

        @njit
        def check(x):
            vs = np.arange(10, dtype=np.float64)
            ctr = 11
            obj = MyStruct(vs, ctr)
            return obj.testme(x)

        x = 3
        got = check(x)
        expect = check.py_func(x)
        self.assertPreciseEqual(got, expect)
