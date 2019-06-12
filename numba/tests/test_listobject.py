""" Tests for the Numba typed list components at:

    * numba/_listobject.h
    * numba/_listobject.c
    * numba/listobject.py

These are the C and compiler components of the typed list implementation. The
tests here should exercise everything within an `@njit` context. Importantly,
the tests should not return a typed list from within such a context as this
would require code from numba/typed/typedlist.py (this is tested seperately).
Tests in this file build on each other in the order of writing. For example,
the first test, tests the creation, append and len of the list. These are the
barebones to do anything useful with a list. The subsequent test for getitem
assumes makes use of these three operations and therefore assumes that they
work.

"""
from __future__ import print_function, absolute_import, division

from numba import njit
from numba import int32, types
from numba import listobject
from numba.utils import IS_PY3
from .support import TestCase, MemoryLeakMixin, unittest

skip_py2 = unittest.skipUnless(IS_PY3, reason='not supported in py2')


class TestListObjectCreateAppendLength(MemoryLeakMixin, TestCase):
    """Test list creation, append and len. """

    def test_list_create(self):
        @njit
        def foo(n):
            l = listobject.new_list(int32)
            for i in range(n):
                l.append(i)
            return len(l)

        for i in (0, 1, 2, 100):
            self.assertEqual(foo(i), i)


class TestListObjectGetitem(MemoryLeakMixin, TestCase):
    """Test list getitem. """

    def test_list_getitem_singleton(self):
        @njit
        def foo(n):
            l = listobject.new_list(int32)
            l.append(n)
            return l[0]

        self.assertEqual(foo(0), 0)

    def test_list_getitem_singleton_negtive_index(self):
        @njit
        def foo(n):
            l = listobject.new_list(int32)
            l.append(n)
            return l[-1]

        self.assertEqual(foo(0), 0)

    def test_list_getitem_multiple(self):
        @njit
        def foo(i):
            l = listobject.new_list(int32)
            for j in range(10, 20):
                l.append(j)
            return l[i]

        for i,j in ((0, 10), (9, 19), (4, 14), (-5, 15), (-1, 19), (-10, 10)):
            self.assertEqual(foo(i), j)

    def test_list_getitem_empty_index_error(self):
        self.disable_leak_check()

        @njit
        def bar(i):
            l = listobject.new_list(int32)
            return l[i]

        with self.assertRaises(IndexError):
            bar(1)

        with self.assertRaises(IndexError):
            bar(0)

        with self.assertRaises(IndexError):
            bar(-1)

    def test_list_getitem_multiple_index_error(self):
        self.disable_leak_check()

        @njit
        def bar(i):
            l = listobject.new_list(int32)
            for j in range(10, 20):
                l.append(j)
            return l[i]

        with self.assertRaises(IndexError):
            bar(10)

        with self.assertRaises(IndexError):
            bar(-11)


class TestListObjectSetitem(MemoryLeakMixin, TestCase):
    """Test list setitem. """

    def test_list_setitem_singleton(self):
        @njit
        def foo(n):
            l = listobject.new_list(int32)
            l.append(0)
            l[0] = n
            return l[0]

        for i in (0, 1, 2, 100):
            self.assertEqual(foo(i), i)

    def test_list_setitem_singleton_negative_index(self):
        @njit
        def foo(n):
            l = listobject.new_list(int32)
            l.append(0)
            l[0] = n
            return l[-1]

        for i in (0, 1, 2, 100):
            self.assertEqual(foo(i), i)

    def test_list_setitem_singleton_index_error(self):
        self.disable_leak_check()

        @njit
        def foo(i):
            l = listobject.new_list(int32)
            l.append(0)
            l[i] = 1

        with self.assertRaises(IndexError):
            foo(1)

        with self.assertRaises(IndexError):
            foo(-2)

    def test_list_setitem_multiple(self):

        @njit
        def foo(i, n):
            l = listobject.new_list(int32)
            for j in range(10, 20):
                l.append(j)
            l[i] = n
            return l[i]

        for i,n in zip(range(0,10), range(20,30)):
            self.assertEqual(foo(i, n), n)

    def test_list_setitem_multiple_index_error(self):
        self.disable_leak_check()

        @njit
        def bar(i):
            l = listobject.new_list(int32)
            for j in range(10, 20):
                l.append(j)
            l[i] = 0

        with self.assertRaises(IndexError):
            bar(10)

        with self.assertRaises(IndexError):
            bar(-11)


class TestListObjectPop(MemoryLeakMixin, TestCase):
    """Test list pop. """

    def test_list_pop_singleton(self):
        @njit
        def foo():
            l = listobject.new_list(int32)
            l.append(0)
            return l.pop(), len(l)

        self.assertEqual(foo(), (0, 0))

    def test_list_pop_singleton_index(self):
        @njit
        def foo(i):
            l = listobject.new_list(int32)
            l.append(0)
            return l.pop(i), len(l)

        self.assertEqual(foo(0), (0, 0))
        self.assertEqual(foo(-1), (0, 0))

    def test_list_pop_multiple(self):
        @njit
        def foo():
            l = listobject.new_list(int32)
            for j in (10, 11, 12):
                l.append(j)
            return l.pop(), len(l)

        self.assertEqual(foo(), (12, 2))

    def test_list_pop_multiple_index(self):
        @njit
        def foo(i):
            l = listobject.new_list(int32)
            for j in (10, 11, 12):
                l.append(j)
            return l.pop(i), len(l)

        for i, n in ((0, 10), (1, 11), (2, 12)):
            self.assertEqual(foo(i), (n, 2))

        for i, n in ((-3, 10), (-2, 11), (-1, 12)):
            self.assertEqual(foo(i), (n, 2))

    def test_list_pop_empty_index_error_no_index(self):
        self.disable_leak_check()

        @njit
        def foo():
            l = listobject.new_list(int32)
            l.pop()

        with self.assertRaises(IndexError):
            foo()

    def test_list_pop_empty_index_error_with_index(self):
        self.disable_leak_check()
        @njit
        def foo(i):
            l = listobject.new_list(int32)
            l.pop(i)

        with self.assertRaises(IndexError):
            foo(-1)

        with self.assertRaises(IndexError):
            foo(0)

        with self.assertRaises(IndexError):
            foo(1)

    def test_list_pop_mutiple_index_error_with_index(self):
        self.disable_leak_check()
        @njit
        def foo(i):
            l = listobject.new_list(int32)
            for j in (10, 11, 12):
                l.append(j)
            l.pop(i)

        with self.assertRaises(IndexError):
            foo(-4)

        with self.assertRaises(IndexError):
            foo(3)


class TestListObjectContains(MemoryLeakMixin, TestCase):
    """Test list contains. """

    def test_list_contains_empty(self):
        @njit
        def foo(i):
            l = listobject.new_list(int32)
            return i in l

        self.assertFalse(foo(0))
        self.assertFalse(foo(1))

    def test_list_contains_singleton(self):
        @njit
        def foo(i):
            l = listobject.new_list(int32)
            l.append(0)
            return i in l

        self.assertTrue(foo(0))
        self.assertFalse(foo(1))

    def test_list_contains_multiple(self):
        @njit
        def foo(i):
            l = listobject.new_list(int32)
            for j in range(10, 20):
                l.append(j)
            return i in l

        for i in range(10, 20):
            self.assertTrue(foo(i))

        for i in range(20, 30):
            self.assertFalse(foo(i))


class TestListObjectCount(MemoryLeakMixin, TestCase):
    """Test list count. """

    def test_list_count_empty(self):
        @njit
        def foo(i):
            l = listobject.new_list(int32)
            return l.count(i)

        self.assertEqual(foo(10), 0)

    def test_list_count_singleton(self):
        @njit
        def foo(i):
            l = listobject.new_list(int32)
            l.append(10)
            return l.count(i)

        self.assertEqual(foo(1), 0)
        self.assertEqual(foo(10), 1)

    def test_list_count_mutiple(self):
        @njit
        def foo(i):
            l = listobject.new_list(int32)
            for j in [11, 12, 12, 13, 13, 13]:
                l.append(j)
            return l.count(i)

        self.assertEqual(foo(10), 0)
        self.assertEqual(foo(11), 1)
        self.assertEqual(foo(12), 2)
        self.assertEqual(foo(13), 3)


class TestListObjectExtend(MemoryLeakMixin, TestCase):
    """Test list extend. """

    def test_list_extend_empty(self):
        @njit
        def foo(items):
            l = listobject.new_list(int32)
            l.extend(items)
            return len(l)

        self.assertEqual(foo((1,)), 1)
        self.assertEqual(foo((1,2)), 2)
        self.assertEqual(foo((1,2,3)), 3)


class TestListObjectInsert(MemoryLeakMixin, TestCase):
    """Test list insert. """

    def test_list_insert_empty(self):
        @njit
        def foo(i):
            l = listobject.new_list(int32)
            l.insert(i, 1)
            return len(l), l[0]

        for i in (-10, -5, -1, 0, 1, 4, 9):
            self.assertEqual(foo(i), (1, 1))

    def test_list_insert_singleton(self):
        @njit
        def foo(i):
            l = listobject.new_list(int32)
            l.append(0)
            l.insert(i, 1)
            return len(l), l[0], l[1]

        # insert before
        for i in (-10, -3, -2, -1, 0):
            self.assertEqual(foo(i), (2, 1, 0))

        # insert after
        for i in (1, 2, 3, 10):
            self.assertEqual(foo(i), (2, 0, 1))

    def test_list_insert_multiple(self):
        @njit
        def foo(i):
            l = listobject.new_list(int32)
            for j in range(10):
                l.append(0)
            l.insert(i, 1)
            return len(l), l[i]

        for i in (0, 4, 9):
            self.assertEqual(foo(i), (11, 1))

    def test_list_insert_multiple_before(self):
        @njit
        def foo(i):
            l = listobject.new_list(int32)
            for j in range(10):
                l.append(0)
            l.insert(i, 1)
            return len(l), l[0]

        for i in (-12, -11, -10, 0):
            self.assertEqual(foo(i), (11, 1))

    def test_list_insert_multiple_after(self):
        @njit
        def foo(i):
            l = listobject.new_list(int32)
            for j in range(10):
                l.append(0)
            l.insert(i, 1)
            return len(l), l[10]

        for i in (10, 11, 12):
            self.assertEqual(foo(i), (11, 1))


class TestListRemove(MemoryLeakMixin, TestCase):
    """Test list remove. """

    def test_list_remove_empty(self):
        self.disable_leak_check()
        @njit
        def foo():
            l = listobject.new_list(int32)
            l.remove(0)

        with self.assertRaises(ValueError):
            foo()

    def test_list_remove_singleton(self):
        @njit
        def foo():
            l = listobject.new_list(int32)
            l.append(0)
            l.remove(0)
            return len(l)

        self.assertEqual(foo(), 0)

    def test_list_remove_singleton_value_error(self):
        self.disable_leak_check()
        @njit
        def foo():
            l = listobject.new_list(int32)
            l.append(1)
            l.remove(0)

        with self.assertRaises(ValueError):
            foo()

    def test_list_remove_multiple(self):
        @njit
        def foo():
            l = listobject.new_list(int32)
            for j in range(10, 20):
                l.append(j)
            l.remove(13)
            l.remove(19)
            return len(l)

        self.assertEqual(foo(), 8)

    def test_list_remove_multiple_value_error(self):
        self.disable_leak_check()
        @njit
        def foo():
            l = listobject.new_list(int32)
            for j in range(10, 20):
                l.append(j)
            l.remove(23)

        with self.assertRaises(ValueError):
            foo()


class TestListObjectIter(MemoryLeakMixin, TestCase):
    """Test list iter. """

    def test_list_iter(self):
        """
        Exercise iter(list)
        """
        @njit
        def foo(items):
            l = listobject.new_list(int32)
            l.extend(items)
            # use a simple sum to check this w/o having to return a list
            r = 0
            for j in l:
                r += j
            return r

        items = (1, 2, 3, 4)

        self.assertEqual(
            foo(items),
            sum(items)
        )


class TestListObjectStringItem(MemoryLeakMixin, TestCase):
    """Test list can take strings as items. """

    @skip_py2
    def test_string_item(self):
        @njit
        def foo():
            l = listobject.new_list(types.unicode_type)
            l.append('a')
            l.append('b')
            l.append('c')
            l.append('d')
            return l[0], l[1], l[2], l[3]

        items = foo()
        self.assertEqual(['a', 'b', 'c', 'd'], list(items))
