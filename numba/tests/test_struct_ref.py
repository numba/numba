"""
Test mutable struct, aka, structref
"""

from types import MappingProxyType

import numpy as np

from numba.core import types
from numba import njit


from numba.core import structref
from numba.tests.support import MemoryLeakMixin, TestCase


class MyStruct(types.Type):
    def __init__(self, typename, fields):
        self._typename = typename
        self._fields = tuple(fields)
        super().__init__(name=f"numba.structref.{self.__class__.__name__}")

    @property
    def fields(self):
        return self._fields

    @property
    def field_dict(self):
        return MappingProxyType(dict(self._fields))

    def get_data_type(self):
        return types.StructPayloadType(
            typename=self._typename, fields=self._fields,
        )


structref.register(MyStruct)

my_struct_ty = MyStruct(
    "MyStruct", fields=[("values", types.intp[:]), ("counter", types.intp)]
)


class MyStructWrap:
    def __init__(self, ty, mi):
        self._ty = ty
        self._mi = mi

    @property
    def _numba_type_(self):
        return self._ty


def object_ctor(ty, mi):
    return MyStructWrap(ty, mi)


structref.define_boxing(MyStruct, object_ctor)


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
def foo(vs, ctr):
    st = my_struct(vs, counter=ctr)
    st.values += st.values
    st.counter *= ctr
    return st


@njit
def get_values(st):
    return st.values


@njit
def bar(st):
    return st.values + st.counter


class TestStructRef(MemoryLeakMixin, TestCase):
    def test_basic(self):
        vs = np.arange(10)
        ctr = 10

        foo_expected = vs + vs
        foo_got = foo(vs, ctr)
        self.assertPreciseEqual(foo_expected, get_values(foo_got))

        bar_expected = foo_expected + (ctr * ctr)
        bar_got = bar(foo_got)
        self.assertPreciseEqual(bar_expected, bar_got)
