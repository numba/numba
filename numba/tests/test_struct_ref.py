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
    def __init__(self, fields):
        self._fields = tuple(fields)
        classname = self.__class__.__name__
        super().__init__(name=f"numba.structref.{classname}{self._fields}")

    @property
    def fields(self):
        return self._fields

    @property
    def field_dict(self):
        return MappingProxyType(dict(self._fields))

    def get_data_type(self):
        return types.StructPayloadType(
            typename=self.__class__.__name__, fields=self._fields,
        )


structref.register(MyStruct)

my_struct_ty = MyStruct(
    fields=[("values", types.intp[:]), ("counter", types.intp)]
)


class MyStructWrap:
    def __init__(self, ty, mi):
        self._ty = ty
        self._mi = mi

    @property
    def _numba_type_(self):
        return self._ty


def _my_struct_wrap_ctor(ty, mi):
    return MyStructWrap(ty, mi)


structref.define_boxing(MyStruct, _my_struct_wrap_ctor)

structref.define_constructor(
    lambda xs: MyStruct(fields=xs),
    MyStructWrap,
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
    return MyStructWrap(values=vs, counter=ctr)


@njit
def get_values(st):
    return st.values


@njit
def compute_fields(st):
    return st.values + st.counter


class TestStructRef(MemoryLeakMixin, TestCase):
    def test_ctor_by_intrinsic(self):
        vs = np.arange(10, dtype=np.intp)
        ctr = 10

        first_expected = vs + vs
        first_got = ctor_by_intrinsic(vs, ctr)
        self.assertPreciseEqual(first_expected, get_values(first_got))

        second_expected = first_expected + (ctr * ctr)
        second_got = compute_fields(first_got)
        self.assertPreciseEqual(second_expected, second_got)

    def test_ctor_by_class(self):
        vs = np.arange(10, dtype=np.float64)
        ctr = 10

        first_expected = vs.copy()
        first_got = ctor_by_class(vs, ctr)
        self.assertPreciseEqual(first_expected, get_values(first_got))

        second_expected = first_expected + ctr
        second_got = compute_fields(first_got)
        self.assertPreciseEqual(second_expected, second_got)
