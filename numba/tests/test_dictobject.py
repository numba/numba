"""
Testing numba implementation of the numba dictionary
"""
from __future__ import print_function, absolute_import, division

import random


from numba import njit
from numba import int32, float32, float64
from numba import dictobject
from .support import TestCase, MemoryLeakMixin


class TestDictObject(MemoryLeakMixin, TestCase):
    def test_dict_create(self):
        """
        Exercise dictionary creation, insertion and len
        """
        @njit
        def foo(n):
            d = dictobject.new_dict(int32, float32)
            for i in range(n):
                d[i] = i + 1
            return len(d)

        # Insert nothing
        self.assertEqual(foo(n=0), 0)
        # Insert 1 entry
        self.assertEqual(foo(n=1), 1)
        # Insert 2 entries
        self.assertEqual(foo(n=2), 2)
        # Insert 100 entries
        self.assertEqual(foo(n=100), 100)

    def test_dict_get(self):
        """
        Exercise dictionary creation, insertion and get
        """
        @njit
        def foo(n, targets):
            d = dictobject.new_dict(int32, float64)
            # insertion loop
            for i in range(n):
                d[i] = i
            # retrieval loop
            output = []
            for t in targets:
                output.append(d.get(t))
            return output

        self.assertEqual(foo(5, [0, 1, 9]), [0, 1, None])
        self.assertEqual(foo(10, [0, 1, 9]), [0, 1, 9])
        self.assertEqual(foo(10, [-1, 9, 1]), [None, 9, 1])

    def test_dict_get_with_default(self):
        """
        Exercise dict.get(k, d) where d is set
        """
        @njit
        def foo(n, target, default):
            d = dictobject.new_dict(int32, float64)
            # insertion loop
            for i in range(n):
                d[i] = i
            # retrieval loop
            return d.get(target, d=default)

        self.assertEqual(foo(5, 3, -1), 3)
        self.assertEqual(foo(5, 5, -1), -1)

    def test_dict_getitem(self):
        """
        Exercise dictionary __getitem__
        """
        @njit
        def foo(keys, vals, target):
            d = dictobject.new_dict(int32, float64)
            # insertion
            for k, v in zip(keys, vals):
                d[k] = v

            # lookup
            return d[target]

        keys = [1, 2, 3]
        vals = [0.1, 0.2, 0.3]
        self.assertEqual(foo(keys, vals, 1), 0.1)
        self.assertEqual(foo(keys, vals, 2), 0.2)
        self.assertEqual(foo(keys, vals, 3), 0.3)

        with self.assertRaises(KeyError):
            foo(keys, vals, 0)
        with self.assertRaises(KeyError):
            foo(keys, vals, 4)

    def test_dict_popitem(self):
        """
        Exercise dictionary .popitem
        """
        @njit
        def foo(keys, vals):
            d = dictobject.new_dict(int32, float64)
            # insertion
            for k, v in zip(keys, vals):
                d[k] = v

            # popitem
            return d.popitem()

        keys = [1, 2, 3]
        vals = [0.1, 0.2, 0.3]
        for i in range(1, len(keys)):
            self.assertEqual(
                foo(keys[:i], vals[:i]),
                (keys[i - 1], vals[i - 1]),
            )

    def test_dict_popitem_many(self):
        """
        Exercise dictionary .popitem
        """

        @njit
        def core(d, npop):
            # popitem
            keysum, valsum = 0, 0
            for _ in range(npop):
                k, v = d.popitem()
                keysum += k
                valsum -= v
            return keysum, valsum

        @njit
        def foo(keys, vals, npop):
            d = dictobject.new_dict(int32, int32)
            # insertion
            for k, v in zip(keys, vals):
                d[k] = v

            return core(d, npop)

        keys = [1, 2, 3]
        vals = [10, 20, 30]

        for i in range(len(keys)):
            self.assertEqual(
                foo(keys, vals, npop=3),
                core.py_func(dict(zip(keys, vals)), npop=3),
            )

        with self.assertRaises(KeyError):
            foo(keys, vals, npop=4)

