"""
Test mutable struct, aka, structref
"""
import numpy as np

from numba.core import types
from numba import njit


from numba.core import structref
from numba.tests.support import MemoryLeakMixin, TestCase


my_struct_ty = types.StructRef(
    fields=[("values", types.intp[:]), ("counter", types.intp)]
)


class MyStruct(structref.StructRefProxy):
    pass


structref.define_constructor(
    # The lambda function here is not necessary, but as a test to show that
    # the first arg can be a callable that makes a StructRef.
    lambda xs: types.StructRef(fields=xs),
    MyStruct,
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
def compute_fields(st):
    return st.values + st.counter


class TestStructRef(MemoryLeakMixin, TestCase):
    def test_ctor_by_intrinsic(self):
        vs = np.arange(10, dtype=np.intp)
        ctr = 13

        first_expected = vs + vs
        first_got = ctor_by_intrinsic(vs, ctr)
        self.assertPreciseEqual(first_expected, get_values(first_got))

        second_expected = first_expected + (ctr * ctr)
        second_got = compute_fields(first_got)
        self.assertPreciseEqual(second_expected, second_got)

    def test_ctor_by_class(self):
        vs = np.arange(10, dtype=np.float64)
        ctr = 11

        first_expected = vs.copy()
        first_got = ctor_by_class(vs, ctr)
        self.assertPreciseEqual(first_expected, get_values(first_got))

        second_expected = first_expected + ctr
        second_got = compute_fields(first_got)
        self.assertPreciseEqual(second_expected, second_got)
