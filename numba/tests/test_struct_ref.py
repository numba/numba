"""
Test mutable struct, aka, structref
"""
import warnings

import numpy as np

from numba.core import types
from numba import njit
from numba.core import structref
from numba.extending import overload_method
from numba.tests.support import (
    MemoryLeakMixin, TestCase, temp_directory, override_config,
)


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


@overload_method(MyStructType, "testme")
def _(self, arg):
    def impl(self, arg):
        return self.values * arg + self.counter
    return impl


class TestStructRefExtending(MemoryLeakMixin, TestCase):
    def test_overload_method(self):
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


def caching_test_make(x, y):
    struct = MyStruct(values=x, counter=y)
    return struct


def caching_test_use(struct, z):
    return struct.testme(z)


class TestStructRefCaching(MemoryLeakMixin, TestCase):
    def setUp(self):
        self._cache_dir = temp_directory(TestStructRefCaching.__name__)
        self._cache_override = override_config('CACHE_DIR', self._cache_dir)
        self._cache_override.__enter__()
        warnings.simplefilter("error")

    def tearDown(self):
        self._cache_override.__exit__(None, None, None)
        warnings.resetwarnings()

    def test_structref_caching(self):
        def assert_cached(stats):
            self.assertEqual(len(stats.cache_hits), 1)
            self.assertEqual(len(stats.cache_misses), 0)

        def assert_not_cached(stats):
            self.assertEqual(len(stats.cache_hits), 0)
            self.assertEqual(len(stats.cache_misses), 1)

        def check(cached):
            check_make = njit(cache=True)(caching_test_make)
            check_use = njit(cache=True)(caching_test_use)
            vs = np.random.random(3)
            ctr = 17
            factor = 3
            st = check_make(vs, ctr)
            got = check_use(st, factor)
            expect = vs * factor + ctr
            self.assertPreciseEqual(got, expect)

            if cached:
                assert_cached(check_make.stats)
                assert_cached(check_use.stats)
            else:
                assert_not_cached(check_make.stats)
                assert_not_cached(check_use.stats)

        check(cached=False)
        check(cached=True)
