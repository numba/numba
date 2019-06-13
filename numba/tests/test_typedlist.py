from __future__ import print_function, absolute_import, division

from itertools import product

from numba import njit
from numba import int32, int64, types
from numba.typed import List, Dict
from numba.utils import IS_PY3
from .support import TestCase, MemoryLeakMixin, unittest

from numba.unsafe.refcount import get_refcount

skip_py2 = unittest.skipUnless(IS_PY3, reason='not supported in py2')


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
        and compare results agains the regular list in a quasi fuzzing approach.

        """
        # initialize regular list
        rl = list(range(10, 20))
        # intialize typed list
        tl = List.empty_list(int32)
        for i in range(10, 20):
            tl.append(i)
        # define the ranges
        start_range = list(range(-10, 30))
        stop_range = list(range(-10, 30))
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
            l = List.empty_list(Dict.empty(int64, int64))
            d = {}
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
            l = List.empty_list(Dict.empty(int64, int64))
            d = {}
            d[0] = 1
            # This increments the refcount for d
            l.append(d)
            l.append(d)
            return get_refcount(d)

        c = foo()
        self.assertEqual(3, c)

    @skip_py2
    def test_list_as_value_in_dict(self):
        @njit
        def foo():
            d = Dict.empty(int64, List.empty_list(int64))
            l = List.empty_list(int64)
            l.append(0)
            # This increments the refcount for l
            d[0] = l
            return get_refcount(l)

        c = foo()
        self.assertEqual(2, c)
