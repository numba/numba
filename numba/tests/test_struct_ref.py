"""
Test mutable struct, aka, structref
"""
import numpy as np

from numba.core import types
from numba import njit
from numba.core import structref
from numba.tests.support import MemoryLeakMixin, TestCase


@structref.register
class MySimplerStructType(types.StructRef):
    pass


my_struct_ty = MySimplerStructType(
    fields=[("values", types.intp[:]), ("counter", types.intp)]
)

structref.define_boxing(MySimplerStructType, structref.StructRefProxy)


class MyStruct(structref.StructRefProxy):
    @property
    def values(self):
        return get_values(self)

    @property
    def counter(self):
        return get_counter(self)


@structref.register
class MyStructType(types.StructRef):
    pass


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


class TestStructRef(MemoryLeakMixin, TestCase):
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
