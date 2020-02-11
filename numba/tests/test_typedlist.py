from __future__ import print_function, absolute_import, division

from itertools import product

import numpy as np

from numba import njit
from numba import int32, float32, types, prange
from numba import jitclass, typeof
from numba.typed import List, Dict
from numba.utils import IS_PY3
from numba.errors import TypingError
from .support import (TestCase, MemoryLeakMixin, unittest, override_config,
                      forbid_codegen)

from numba.unsafe.refcount import get_refcount

from .test_parfors import skip_unsupported as parfors_skip_unsupported

skip_py2 = unittest.skipUnless(IS_PY3, reason='not supported in py2')

# global typed-list for testing purposes
global_typed_list = List.empty_list(int32)
for i in (1, 2, 3):
    global_typed_list.append(int32(i))


def to_tl(l):
    """ Convert cpython list to typed-list. """
    tl = List.empty_list(int32)
    for k in l:
        tl.append(k)
    return tl


class TestTypedList(MemoryLeakMixin, TestCase):
    def test_basic(self):
        l = List.empty_list(int32)
        # len
        self.assertEqual(len(l), 0)
        # append
        l.append(0)
        # len
        self.assertEqual(len(l), 1)
        # setitem
        l.append(0)
        l.append(0)
        l[0] = 10
        l[1] = 11
        l[2] = 12
        # getitem
        self.assertEqual(l[0], 10)
        self.assertEqual(l[1], 11)
        self.assertEqual(l[2], 12)
        self.assertEqual(l[-3], 10)
        self.assertEqual(l[-2], 11)
        self.assertEqual(l[-1], 12)
        # __iter__
        # the default __iter__ from MutableSequence will raise an IndexError
        # via __getitem__ and thus leak an exception, so this shouldn't
        for i in l:
            pass
        # contains
        self.assertTrue(10 in l)
        self.assertFalse(0 in l)
        # count
        l.append(12)
        self.assertEqual(l.count(0), 0)
        self.assertEqual(l.count(10), 1)
        self.assertEqual(l.count(12), 2)
        # pop
        self.assertEqual(len(l), 4)
        self.assertEqual(l.pop(), 12)
        self.assertEqual(len(l), 3)
        self.assertEqual(l.pop(1), 11)
        self.assertEqual(len(l), 2)
        # extend
        l.extend((100, 200, 300))
        self.assertEqual(len(l), 5)
        self.assertEqual(list(l), [10, 12, 100, 200, 300])
        # insert
        l.insert(0, 0)
        self.assertEqual(list(l), [0, 10, 12, 100, 200, 300])
        l.insert(3, 13)
        self.assertEqual(list(l), [0, 10, 12, 13, 100, 200, 300])
        l.insert(100, 400)
        self.assertEqual(list(l), [0, 10, 12, 13, 100, 200, 300, 400])
        # remove
        l.remove(0)
        l.remove(400)
        l.remove(13)
        self.assertEqual(list(l), [10, 12, 100, 200, 300])
        # clear
        l.clear()
        self.assertEqual(len(l), 0)
        self.assertEqual(list(l), [])
        # reverse
        l.extend(tuple(range(10, 20)))
        l.reverse()
        self.assertEqual(list(l), list(range(10, 20))[::-1])
        # copy
        new = l.copy()
        self.assertEqual(list(new), list(range(10, 20))[::-1])
        # equal
        self.assertEqual(l, new)
        # not equal
        new[-1] = 42
        self.assertNotEqual(l, new)
        # index
        self.assertEqual(l.index(15), 4)

    def test_unsigned_access(self):
        L = List.empty_list(int32)
        ui32_0 = types.uint32(0)
        ui32_1 = types.uint32(1)
        ui32_2 = types.uint32(2)

        # insert
        L.append(types.uint32(10))
        L.append(types.uint32(11))
        L.append(types.uint32(12))
        self.assertEqual(len(L), 3)

        # getitem
        self.assertEqual(L[ui32_0], 10)
        self.assertEqual(L[ui32_1], 11)
        self.assertEqual(L[ui32_2], 12)

        # setitem
        L[ui32_0] = 123
        L[ui32_1] = 456
        L[ui32_2] = 789
        self.assertEqual(L[ui32_0], 123)
        self.assertEqual(L[ui32_1], 456)
        self.assertEqual(L[ui32_2], 789)

        # index
        ui32_123 = types.uint32(123)
        ui32_456 = types.uint32(456)
        ui32_789 = types.uint32(789)
        self.assertEqual(L.index(ui32_123), 0)
        self.assertEqual(L.index(ui32_456), 1)
        self.assertEqual(L.index(ui32_789), 2)

        # delitem
        L.__delitem__(ui32_2)
        del L[ui32_1]
        self.assertEqual(len(L), 1)
        self.assertEqual(L[ui32_0], 123)

        # pop
        L.append(2)
        L.append(3)
        L.append(4)
        self.assertEqual(len(L), 4)
        self.assertEqual(L.pop(), 4)
        self.assertEqual(L.pop(ui32_2), 3)
        self.assertEqual(L.pop(ui32_1), 2)
        self.assertEqual(L.pop(ui32_0), 123)

    @parfors_skip_unsupported
    def test_unsigned_prange(self):
        @njit(parallel=True)
        def foo(a):
            r = types.uint64(3)
            s = types.uint64(0)
            for i in prange(r):
                s = s + a[i]
            return s

        a = List.empty_list(types.uint64)
        a.append(types.uint64(12))
        a.append(types.uint64(1))
        a.append(types.uint64(7))
        self.assertEqual(foo(a), 20)

    def test_compiled(self):
        @njit
        def producer():
            l = List.empty_list(int32)
            l.append(23)
            return l

        @njit
        def consumer(l):
            return l[0]

        l = producer()
        val = consumer(l)
        self.assertEqual(val, 23)

    def test_getitem_slice(self):
        """ Test getitem using a slice.

        This tests suffers from combinatorial explosion, so we parametrize it
        and compare results against the regular list in a quasi fuzzing
        approach.

        """
        # initialize regular list
        rl = list(range(10, 20))
        # initialize typed list
        tl = List.empty_list(int32)
        for i in range(10, 20):
            tl.append(i)
        # define the ranges
        start_range = list(range(-20, 30))
        stop_range = list(range(-20, 30))
        step_range = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]

        # check that they are the same initially
        self.assertEqual(rl, list(tl))
        # check that copy by slice works, no start, no stop, no step
        self.assertEqual(rl[:], list(tl[:]))

        # start only
        for sa in start_range:
            self.assertEqual(rl[sa:], list(tl[sa:]))
        # stop only
        for so in stop_range:
            self.assertEqual(rl[:so], list(tl[:so]))
        # step only
        for se in step_range:
            self.assertEqual(rl[::se], list(tl[::se]))

        # start and stop
        for sa, so in product(start_range, stop_range):
            self.assertEqual(rl[sa:so], list(tl[sa:so]))
        # start and step
        for sa, se in product(start_range, step_range):
            self.assertEqual(rl[sa::se], list(tl[sa::se]))
        # stop and step
        for so, se in product(stop_range, step_range):
            self.assertEqual(rl[:so:se], list(tl[:so:se]))

        # start, stop and step
        for sa, so, se in product(start_range, stop_range, step_range):
            self.assertEqual(rl[sa:so:se], list(tl[sa:so:se]))

    def test_setitem_slice(self):
        """ Test setitem using a slice.

        This tests suffers from combinatorial explosion, so we parametrize it
        and compare results against the regular list in a quasi fuzzing
        approach.

        """

        def setup(start=10, stop=20):
            # initialize regular list
            rl_ = list(range(start, stop))
            # initialize typed list
            tl_ = List.empty_list(int32)
            # populate typed list
            for i in range(start, stop):
                tl_.append(i)
            # check they are the same
            self.assertEqual(rl_, list(tl_))
            return rl_, tl_

        ### Simple slicing ###

        # assign to itself
        rl, tl = setup()
        rl[:], tl[:] = rl, tl
        self.assertEqual(rl, list(tl))

        # extend self
        rl, tl = setup()
        rl[len(rl):], tl[len(tl):] = rl, tl
        self.assertEqual(rl, list(tl))
        # prepend self
        rl, tl = setup()
        rl[:0], tl[:0] = rl, tl
        self.assertEqual(rl, list(tl))
        # partial assign to self, with equal length
        rl, tl = setup()
        rl[3:5], tl[3:5] = rl[6:8], tl[6:8]
        self.assertEqual(rl, list(tl))
        # partial assign to self, with larger slice
        rl, tl = setup()
        rl[3:5], tl[3:5] = rl[6:9], tl[6:9]
        self.assertEqual(rl, list(tl))
        # partial assign to self, with smaller slice
        rl, tl = setup()
        rl[3:5], tl[3:5] = rl[6:7], tl[6:7]
        self.assertEqual(rl, list(tl))

        # extend
        rl, tl = setup()
        rl[len(rl):] = list(range(110, 120))
        tl[len(tl):] = to_tl(range(110,120))
        self.assertEqual(rl, list(tl))
        # extend empty
        rl, tl = setup(0, 0)
        rl[len(rl):] = list(range(110, 120))
        tl[len(tl):] = to_tl(range(110,120))
        self.assertEqual(rl, list(tl))
        # extend singleton
        rl, tl = setup(0, 1)
        rl[len(rl):] = list(range(110, 120))
        tl[len(tl):] = to_tl(range(110,120))
        self.assertEqual(rl, list(tl))

        # prepend
        rl, tl = setup()
        rl[:0], tl[:0] = list(range(110, 120)), to_tl(range(110,120))
        self.assertEqual(rl, list(tl))
        # prepend empty
        rl, tl = setup(0,0)
        rl[:0], tl[:0] = list(range(110, 120)), to_tl(range(110,120))
        self.assertEqual(rl, list(tl))
        # prepend singleton
        rl, tl = setup(0,1)
        rl[:0], tl[:0] = list(range(110, 120)), to_tl(range(110,120))
        self.assertEqual(rl, list(tl))

        # simple equal length assignment, just replace
        rl, tl = setup()
        rl[1:3], tl[1:3] = [100, 200], to_tl([100, 200])
        self.assertEqual(rl, list(tl))

        # slice for assignment is larger, need to replace and insert
        rl, tl = setup()
        rl[1:3], tl[1:3] = [100, 200, 300, 400], to_tl([100, 200, 300, 400])
        self.assertEqual(rl, list(tl))

        # slice for assignment is smaller, need to replace and delete
        rl, tl = setup()
        rl[1:3], tl[1:3] = [100], to_tl([100])
        self.assertEqual(rl, list(tl))

        # slice for assignment is smaller and item is empty, need to delete
        rl, tl = setup()
        rl[1:3], tl[1:3] = [], to_tl([])
        self.assertEqual(rl, list(tl))

        # Synonym for clear
        rl, tl = setup()
        rl[:], tl[:] = [], to_tl([])
        self.assertEqual(rl, list(tl))

        ### Extended slicing ###

        # replace every second element
        rl, tl = setup()
        rl[::2], tl[::2] = [100,200,300,400,500], to_tl([100,200,300,400,500])
        self.assertEqual(rl, list(tl))
        # replace every second element, backwards
        rl, tl = setup()
        rl[::-2], tl[::-2] = [100,200,300,400,500], to_tl([100,200,300,400,500])
        self.assertEqual(rl, list(tl))

        # reverse assign to itself
        rl, tl = setup()
        rl[::-1], tl[::-1] = rl, tl
        self.assertEqual(rl, list(tl))

    def test_setitem_slice_value_error(self):
        self.disable_leak_check()

        tl = List.empty_list(int32)
        for i in range(10,20):
            tl.append(i)

        assignment = List.empty_list(int32)
        for i in range(1, 4):
            assignment.append(i)

        with self.assertRaises(ValueError) as raises:
            tl[8:3:-1] = assignment
        self.assertIn(
            "length mismatch for extended slice and sequence",
            str(raises.exception),
        )

    def test_delitem_slice(self):
        """ Test delitem using a slice.

        This tests suffers from combinatorial explosion, so we parametrize it
        and compare results against the regular list in a quasi fuzzing
        approach.

        """

        def setup(start=10, stop=20):
            # initialize regular list
            rl_ = list(range(start, stop))
            # initialize typed list
            tl_ = List.empty_list(int32)
            # populate typed list
            for i in range(start, stop):
                tl_.append(i)
            # check they are the same
            self.assertEqual(rl_, list(tl_))
            return rl_, tl_

        # define the ranges
        start_range = list(range(-20, 30))
        stop_range = list(range(-20, 30))
        step_range = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]

        rl, tl = setup()
        # check that they are the same initially
        self.assertEqual(rl, list(tl))
        # check that deletion of the whole list by slice works
        del rl[:]
        del tl[:]
        self.assertEqual(rl, list(tl))

        # start only
        for sa in start_range:
            rl, tl = setup()
            del rl[sa:]
            del tl[sa:]
            self.assertEqual(rl, list(tl))
        # stop only
        for so in stop_range:
            rl, tl = setup()
            del rl[:so]
            del tl[:so]
            self.assertEqual(rl, list(tl))
        # step only
        for se in step_range:
            rl, tl = setup()
            del rl[::se]
            del tl[::se]
            self.assertEqual(rl, list(tl))

        # start and stop
        for sa, so in product(start_range, stop_range):
            rl, tl = setup()
            del rl[sa:so]
            del tl[sa:so]
            self.assertEqual(rl, list(tl))
        # start and step
        for sa, se in product(start_range, step_range):
            rl, tl = setup()
            del rl[sa::se]
            del tl[sa::se]
            self.assertEqual(rl, list(tl))
        # stop and step
        for so, se in product(stop_range, step_range):
            rl, tl = setup()
            del rl[:so:se]
            del tl[:so:se]
            self.assertEqual(rl, list(tl))

        # start, stop and step
        for sa, so, se in product(start_range, stop_range, step_range):
            rl, tl = setup()
            del rl[sa:so:se]
            del tl[sa:so:se]
            self.assertEqual(rl, list(tl))

    def test_list_create_no_jit_using_empty_list(self):
        with override_config('DISABLE_JIT', True):
            with forbid_codegen():
                l = List.empty_list(types.int32)
                self.assertEqual(type(l), list)

    def test_list_create_no_jit_using_List(self):
        with override_config('DISABLE_JIT', True):
            with forbid_codegen():
                l = List()
                self.assertEqual(type(l), list)

    def test_catch_global_typed_list(self):
        @njit()
        def foo():
            x = List()
            for i in global_typed_list:
                x.append(i)

        expected_message = ("The use of a ListType[int32] type, assigned to "
                            "variable 'global_typed_list' in globals, is not "
                            "supported as globals are considered compile-time "
                            "constants and there is no known way to compile "
                            "a ListType[int32] type as a constant.")
        with self.assertRaises(TypingError) as raises:
            foo()
        self.assertIn(
            expected_message,
            str(raises.exception),
        )


class TestAllocation(MemoryLeakMixin, TestCase):

    def test_allocation(self):
        # kwarg version
        for i in range(16):
            tl = List.empty_list(types.int32, allocated=i)
            self.assertEqual(tl._allocated(), i)

        # posarg version
        for i in range(16):
            tl = List.empty_list(types.int32, i)
            self.assertEqual(tl._allocated(), i)

    def test_growth_and_shrinkage(self):
        tl = List.empty_list(types.int32)
        growth_before = {0: 0, 4:4, 8:8, 16:16}
        growth_after = {0: 4, 4:8, 8:16, 16:25}
        for i in range(17):
            if i in growth_before:
                self.assertEqual(growth_before[i], tl._allocated())
            tl.append(i)
            if i in growth_after:
                self.assertEqual(growth_after[i], tl._allocated())

        shrink_before = {17: 25, 12:25, 9:18, 6:12, 4:8, 3:6, 2:5, 1:4}
        shrink_after = {17: 25, 12:18, 9:12, 6:8, 4:6, 3:5, 2:4, 1:0}
        for i in range(17, 0, -1):
            if i in shrink_before:
                self.assertEqual(shrink_before[i], tl._allocated())
            tl.pop()
            if i in shrink_after:
                self.assertEqual(shrink_after[i], tl._allocated())


class TestExtend(MemoryLeakMixin, TestCase):

    def test_extend_other(self):
        @njit
        def impl(other):
            l = List.empty_list(types.int32)
            for x in range(10):
                l.append(x)
            l.extend(other)
            return l

        other = List.empty_list(types.int32)
        for x in range(10):
            other.append(x)

        expected = impl.py_func(other)
        got = impl(other)
        self.assertEqual(expected, got)

    def test_extend_self(self):
        @njit
        def impl():
            l = List.empty_list(types.int32)
            for x in range(10):
                l.append(x)
            l.extend(l)
            return l

        expected = impl.py_func()
        got = impl()
        self.assertEqual(expected, got)

    def test_extend_tuple(self):
        @njit
        def impl():
            l = List.empty_list(types.int32)
            for x in range(10):
                l.append(x)
            l.extend((100,200,300))
            return l

        expected = impl.py_func()
        got = impl()
        self.assertEqual(expected, got)


@njit
def cmp(a, b):
    return a < b, a <= b, a == b, a != b, a >= b, a > b


class TestComparisons(MemoryLeakMixin, TestCase):

    def _cmp_dance(self, expected, pa, pb, na, nb):
        # interpreter with regular list
        self.assertEqual(cmp.py_func(pa, pb), expected)

        # interpreter with typed-list
        py_got = cmp.py_func(na, nb)
        self.assertEqual(py_got, expected)

        # compiled with typed-list
        jit_got = cmp(na, nb)
        self.assertEqual(jit_got, expected)

    def test_empty_vs_empty(self):
        pa, pb = [], []
        na, nb = to_tl(pa), to_tl(pb)
        expected = False, True, True, False, True, False
        self._cmp_dance(expected, pa, pb, na, nb)

    def test_empty_vs_singleton(self):
        pa, pb = [], [0]
        na, nb = to_tl(pa), to_tl(pb)
        expected = True, True, False, True, False, False
        self._cmp_dance(expected, pa, pb, na, nb)

    def test_singleton_vs_empty(self):
        pa, pb = [0], []
        na, nb = to_tl(pa), to_tl(pb)
        expected = False, False, False, True, True, True
        self._cmp_dance(expected, pa, pb, na, nb)

    def test_singleton_vs_singleton_equal(self):
        pa, pb = [0], [0]
        na, nb = to_tl(pa), to_tl(pb)
        expected = False, True, True, False, True, False
        self._cmp_dance(expected, pa, pb, na, nb)

    def test_singleton_vs_singleton_less_than(self):
        pa, pb = [0], [1]
        na, nb = to_tl(pa), to_tl(pb)
        expected = True, True, False, True, False, False
        self._cmp_dance(expected, pa, pb, na, nb)

    def test_singleton_vs_singleton_greater_than(self):
        pa, pb = [1], [0]
        na, nb = to_tl(pa), to_tl(pb)
        expected = False, False, False, True, True, True
        self._cmp_dance(expected, pa, pb, na, nb)

    def test_equal(self):
        pa, pb = [1, 2, 3], [1, 2, 3]
        na, nb = to_tl(pa), to_tl(pb)
        expected = False, True, True, False, True, False
        self._cmp_dance(expected, pa, pb, na, nb)

    def test_first_shorter(self):
        pa, pb = [1, 2], [1, 2, 3]
        na, nb = to_tl(pa), to_tl(pb)
        expected = True, True, False, True, False, False
        self._cmp_dance(expected, pa, pb, na, nb)

    def test_second_shorter(self):
        pa, pb = [1, 2, 3], [1, 2]
        na, nb = to_tl(pa), to_tl(pb)
        expected = False, False, False, True, True, True
        self._cmp_dance(expected, pa, pb, na, nb)

    def test_first_less_than(self):
        pa, pb = [1, 2, 2], [1, 2, 3]
        na, nb = to_tl(pa), to_tl(pb)
        expected = True, True, False, True, False, False
        self._cmp_dance(expected, pa, pb, na, nb)

    def test_first_greater_than(self):
        pa, pb = [1, 2, 3], [1, 2, 2]
        na, nb = to_tl(pa), to_tl(pb)
        expected = False, False, False, True, True, True
        self._cmp_dance(expected, pa, pb, na, nb)

    def test_equals_non_list(self):
        l = to_tl([1, 2, 3])
        self.assertFalse(any(cmp.py_func(l, 1)))
        self.assertFalse(any(cmp(l, 1)))


class TestListInferred(TestCase):

    def test_simple_refine_append(self):
        @njit
        def foo():
            l = List()
            l.append(1)
            return l

        expected = foo.py_func()
        got = foo()
        self.assertEqual(expected, got)
        self.assertEqual(list(got), [1])
        self.assertEqual(typeof(got).item_type, typeof(1))

    def test_simple_refine_insert(self):
        @njit
        def foo():
            l = List()
            l.insert(0, 1)
            return l

        expected = foo.py_func()
        got = foo()
        self.assertEqual(expected, got)
        self.assertEqual(list(got), [1])
        self.assertEqual(typeof(got).item_type, typeof(1))

    def test_refine_extend_list(self):
        @njit
        def foo():
            a = List()
            b = List()
            for i in range(3):
                b.append(i)
            a.extend(b)
            return a

        expected = foo.py_func()
        got = foo()
        self.assertEqual(expected, got)
        self.assertEqual(list(got), [0, 1, 2])
        self.assertEqual(typeof(got).item_type, typeof(1))

    def test_refine_extend_set(self):
        @njit
        def foo():
            l = List()
            l.extend((0, 1, 2))
            return l

        expected = foo.py_func()
        got = foo()
        self.assertEqual(expected, got)
        self.assertEqual(list(got), [0, 1, 2])
        self.assertEqual(typeof(got).item_type, typeof(1))

    def test_refine_list_extend_iter(self):
        @njit
        def foo():
            l = List()
            d = Dict()
            d[0] = 0
            # d.keys() provides a DictKeysIterableType
            l.extend(d.keys())
            return l

        got = foo()
        self.assertEqual(0, got[0])


class TestListRefctTypes(MemoryLeakMixin, TestCase):

    @skip_py2
    def test_str_item(self):
        @njit
        def foo():
            l = List.empty_list(types.unicode_type)
            for s in ("a", "ab", "abc", "abcd"):
                l.append(s)
            return l

        l = foo()
        expected = ["a", "ab", "abc", "abcd"]
        for i, s in enumerate(expected):
            self.assertEqual(l[i], s)
        self.assertEqual(list(l), expected)
        # Test insert replacement
        l[3] = 'uxyz'
        self.assertEqual(l[3], 'uxyz')
        # Test list growth
        nelem = 100
        for i in range(4, nelem):
            l.append(str(i))
            self.assertEqual(l[i], str(i))

    @skip_py2
    def test_str_item_refcount_replace(self):
        @njit
        def foo():
            # use some tricks to make ref-counted unicode
            i, j = 'ab', 'c'
            a = i + j
            m, n = 'zy', 'x'
            z = m + n
            l = List.empty_list(types.unicode_type)
            l.append(a)
            # This *should* dec' a and inc' z thus tests that items that are
            # replaced are also dec'ed.
            l[0] = z
            ra, rz = get_refcount(a), get_refcount(z)
            return l, ra, rz

        l, ra, rz = foo()
        self.assertEqual(l[0], "zyx")
        self.assertEqual(ra, 1)
        self.assertEqual(rz, 2)

    @skip_py2
    def test_dict_as_item_in_list(self):
        @njit
        def foo():
            l = List.empty_list(Dict.empty(int32, int32))
            d = Dict.empty(int32, int32)
            d[0] = 1
            # This increments the refcount for d
            l.append(d)
            return get_refcount(d)

        c = foo()
        self.assertEqual(2, c)

    @skip_py2
    def test_dict_as_item_in_list_multi_refcount(self):
        @njit
        def foo():
            l = List.empty_list(Dict.empty(int32, int32))
            d = Dict.empty(int32, int32)
            d[0] = 1
            # This increments the refcount for d, twice
            l.append(d)
            l.append(d)
            return get_refcount(d)

        c = foo()
        self.assertEqual(3, c)

    @skip_py2
    def test_list_as_value_in_dict(self):
        @njit
        def foo():
            d = Dict.empty(int32, List.empty_list(int32))
            l = List.empty_list(int32)
            l.append(0)
            # This increments the refcount for l
            d[0] = l
            return get_refcount(l)

        c = foo()
        self.assertEqual(2, c)

    @skip_py2
    def test_list_as_item_in_list(self):
        nested_type = types.ListType(types.int32)
        @njit
        def foo():
            la = List.empty_list(nested_type)
            lb = List.empty_list(types.int32)
            lb.append(1)
            la.append(lb)
            return la

        expected = foo.py_func()
        got = foo()
        self.assertEqual(expected, got)

    @skip_py2
    def test_array_as_item_in_list(self):
        nested_type = types.Array(types.float64, 1, 'C')
        @njit
        def foo():
            l = List.empty_list(nested_type)
            a = np.zeros((1,))
            l.append(a)
            return l

        expected = foo.py_func()
        got = foo()
        # Need to compare the nested arrays
        self.assertTrue(np.all(expected[0] == got[0]))

    @skip_py2
    def test_jitclass_as_item_in_list(self):

        spec = [
            ('value', int32),               # a simple scalar field
            ('array', float32[:]),          # an array field
        ]

        @jitclass(spec)
        class Bag(object):
            def __init__(self, value):
                self.value = value
                self.array = np.zeros(value, dtype=np.float32)

            @property
            def size(self):
                return self.array.size

            def increment(self, val):
                for i in range(self.size):
                    self.array[i] += val
                return self.array

        @njit
        def foo():
            l = List()
            l.append(Bag(21))
            l.append(Bag(22))
            l.append(Bag(23))
            return l

        expected = foo.py_func()
        got = foo()

        def bag_equal(one, two):
            # jitclasses couldn't override __eq__ at time of writing
            self.assertEqual(one.value, two.value)
            np.testing.assert_allclose(one.array, two.array)

        [bag_equal(a, b) for a, b in zip(expected, got)]

    @skip_py2
    def test_storage_model_mismatch(self):
        # https://github.com/numba/numba/issues/4520
        # check for storage model mismatch in refcount ops generation
        lst = List()
        ref = [
            ("a", True, "a"),
            ("b", False, "b"),
            ("c", False, "c"),
        ]
        # populate
        for x in ref:
            lst.append(x)
        # test
        for i, x in enumerate(ref):
            self.assertEqual(lst[i], ref[i])

    @skip_py2
    def test_equals_on_list_with_dict_for_equal_lists(self):
        # https://github.com/numba/numba/issues/4879
        a, b = List(), Dict()
        b["a"] = 1
        a.append(b)

        c, d = List(), Dict()
        d["a"] = 1
        c.append(d)

        self.assertEqual(a, c)

    @skip_py2
    def test_equals_on_list_with_dict_for_unequal_dicts(self):
        # https://github.com/numba/numba/issues/4879
        a, b = List(), Dict()
        b["a"] = 1
        a.append(b)

        c, d = List(), Dict()
        d["a"] = 2
        c.append(d)

        self.assertNotEqual(a, c)

    @skip_py2
    def test_equals_on_list_with_dict_for_unequal_lists(self):
        # https://github.com/numba/numba/issues/4879
        a, b = List(), Dict()
        b["a"] = 1
        a.append(b)

        c, d, e = List(), Dict(), Dict()
        d["a"] = 1
        e["b"] = 2
        c.append(d)
        c.append(e)

        self.assertNotEqual(a, c)


class TestListSort(MemoryLeakMixin, TestCase):
    def setUp(self):
        super(TestListSort, self).setUp()
        np.random.seed(0)

    def make(self, ctor, data):
        lst = ctor()
        lst.extend(data)
        return lst

    def make_both(self, data):
        return {
            'py': self.make(list, data),
            'nb': self.make(List, data),
        }

    def test_sort_no_args(self):
        def udt(lst):
            lst.sort()
            return lst

        for nelem in [13, 29, 127]:
            my_lists = self.make_both(np.random.randint(0, nelem, nelem))
            self.assertEqual(list(udt(my_lists['nb'])), udt(my_lists['py']))

    def test_sort_all_args(self):
        def udt(lst, key, reverse):
            lst.sort(key=key, reverse=reverse)
            return lst

        possible_keys = [
            lambda x: -x,           # negative
            lambda x: 1 / (1 + x),  # make float
            lambda x: (x, -x),      # tuple
            lambda x: x,            # identity
        ]
        possible_reverse = [True, False]
        for key, reverse in product(possible_keys, possible_reverse):
            my_lists = self.make_both(np.random.randint(0, 100, 23))
            msg = "case for key={} reverse={}".format(key, reverse)
            self.assertEqual(
                list(udt(my_lists['nb'], key=key, reverse=reverse)),
                udt(my_lists['py'], key=key, reverse=reverse),
                msg=msg,
            )

    def test_sort_dispatcher_key(self):
        def udt(lst, key):
            lst.sort(key=key)
            return lst

        my_lists = self.make_both(np.random.randint(0, 100, 31))
        py_key = lambda x: x + 1
        nb_key = njit(lambda x: x + 1)
        # test typedlist with jitted function
        self.assertEqual(
            list(udt(my_lists['nb'], key=nb_key)),
            udt(my_lists['py'], key=py_key),
        )
        # test typedlist with and without jitted function
        self.assertEqual(
            list(udt(my_lists['nb'], key=nb_key)),
            list(udt(my_lists['nb'], key=py_key)),
        )

    def test_sort_in_jit_w_lambda_key(self):
        @njit
        def udt(lst):
            lst.sort(key=lambda x: -x)
            return lst

        lst = self.make(List, np.random.randint(0, 100, 31))
        self.assertEqual(udt(lst), udt.py_func(lst))

    def test_sort_in_jit_w_global_key(self):
        @njit
        def keyfn(x):
            return -x

        @njit
        def udt(lst):
            lst.sort(key=keyfn)
            return lst

        lst = self.make(List, np.random.randint(0, 100, 31))
        self.assertEqual(udt(lst), udt.py_func(lst))

    def test_sort_on_arrays(self):
        @njit
        def foo(lst):
            lst.sort(key=lambda arr: np.sum(arr))
            return lst

        arrays = [np.random.random(3) for _ in range(10)]
        my_lists = self.make_both(arrays)
        self.assertEqual(
            list(foo(my_lists['nb'])),
            foo.py_func(my_lists['py']),
        )
