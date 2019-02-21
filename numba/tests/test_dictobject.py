"""
Testing numba implementation of the numba dictionary.

The tests here only checks that the numba typing and codegen are working
correctly.  Detail testing of the underlying dictionary operations is done
in test_dictimpl.py.
"""
from __future__ import print_function, absolute_import, division

from numba import njit
from numba import int32, int64, float32, float64
from numba import dictobject
from numba import types
from numba.nbdict import NBDict
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
        # check no leak so far
        self.assert_no_memory_leak()
        # disable leak check for exception test
        self.disable_leak_check()
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

        # check no leak so far
        self.assert_no_memory_leak()
        # disable leak check for exception test
        self.disable_leak_check()

        with self.assertRaises(KeyError):
            foo(keys, vals, npop=4)

    def test_dict_pop(self):
        """
        Exercise dictionary .pop
        """
        @njit
        def foo(keys, vals, target):
            d = dictobject.new_dict(int32, float64)
            # insertion
            for k, v in zip(keys, vals):
                d[k] = v

            # popitem
            return d.pop(target, None), len(d)

        keys = [1, 2, 3]
        vals = [0.1, 0.2, 0.3]

        self.assertEqual(foo(keys, vals, 1), (0.1, 2))
        self.assertEqual(foo(keys, vals, 2), (0.2, 2))
        self.assertEqual(foo(keys, vals, 3), (0.3, 2))
        self.assertEqual(foo(keys, vals, 0), (None, 3))

        # check no leak so far
        self.assert_no_memory_leak()
        # disable leak check for exception test
        self.disable_leak_check()

        @njit
        def foo():
            d = dictobject.new_dict(int32, float64)
            # popitem
            return d.pop(0)

        with self.assertRaises(KeyError):
            foo()

    def test_dict_pop_many(self):
        """
        Exercise dictionary .pop
        """

        @njit
        def core(d, pops):
            total = 0
            for k in pops:
                total += k + d.pop(k, 0.123) + len(d)
                total *= 2
            return total

        @njit
        def foo(keys, vals, pops):
            d = dictobject.new_dict(int32, float64)
            # insertion
            for k, v in zip(keys, vals):
                d[k] = v
            # popitem
            return core(d, pops)

        keys = [1, 2, 3]
        vals = [0.1, 0.2, 0.3]
        pops = [2, 3, 3, 1, 0, 2, 1, 0, -1]

        self.assertEqual(
            foo(keys, vals, pops),
            core.py_func(dict(zip(keys, vals)), pops),
        )

    def test_dict_delitem(self):
        @njit
        def foo(keys, vals, target):
            d = dictobject.new_dict(int32, float64)
            # insertion
            for k, v in zip(keys, vals):
                d[k] = v
            del d[target]
            return len(d), d.get(target)

        keys = [1, 2, 3]
        vals = [0.1, 0.2, 0.3]
        self.assertEqual(foo(keys, vals, 1), (2, None))
        self.assertEqual(foo(keys, vals, 2), (2, None))
        self.assertEqual(foo(keys, vals, 3), (2, None))
        # check no leak so far
        self.assert_no_memory_leak()
        # disable leak check for exception test
        self.disable_leak_check()
        with self.assertRaises(KeyError):
            foo(keys, vals, 0)

    def test_dict_clear(self):
        """
        Exercise dict.clear
        """
        @njit
        def foo(keys, vals):
            d = dictobject.new_dict(int32, float64)
            # insertion
            for k, v in zip(keys, vals):
                d[k] = v
            b4 = len(d)
            # clear
            d.clear()
            return b4, len(d)

        keys = [1, 2, 3]
        vals = [0.1, 0.2, 0.3]
        self.assertEqual(foo(keys, vals), (3, 0))

    def test_dict_items(self):
        """
        Exercise dict.items
        """
        @njit
        def foo(keys, vals):
            d = dictobject.new_dict(int32, float64)
            # insertion
            for k, v in zip(keys, vals):
                d[k] = v
            out = []
            for kv in d.items():
                out.append(kv)
            return out

        keys = [1, 2, 3]
        vals = [0.1, 0.2, 0.3]

        self.assertEqual(
            foo(keys, vals),
            list(zip(keys, vals)),
        )

        # Test .items() on empty dict
        @njit
        def foo():
            d = dictobject.new_dict(int32, float64)
            out = []
            for kv in d.items():
                out.append(kv)
            return out

        self.assertEqual(foo(), [])

    def test_dict_keys(self):
        """
        Exercise dict.keys
        """
        @njit
        def foo(keys, vals):
            d = dictobject.new_dict(int32, float64)
            # insertion
            for k, v in zip(keys, vals):
                d[k] = v
            out = []
            for k in d.keys():
                out.append(k)
            return out

        keys = [1, 2, 3]
        vals = [0.1, 0.2, 0.3]

        self.assertEqual(
            foo(keys, vals),
            keys,
        )

    def test_dict_values(self):
        """
        Exercise dict.values
        """
        @njit
        def foo(keys, vals):
            d = dictobject.new_dict(int32, float64)
            # insertion
            for k, v in zip(keys, vals):
                d[k] = v
            out = []
            for v in d.values():
                out.append(v)
            return out

        keys = [1, 2, 3]
        vals = [0.1, 0.2, 0.3]

        self.assertEqual(
            foo(keys, vals),
            vals,
        )

    def test_dict_iter(self):
        """
        Exercise iter(dict)
        """
        @njit
        def foo(keys, vals):
            d = dictobject.new_dict(int32, float64)
            # insertion
            for k, v in zip(keys, vals):
                d[k] = v
            out = []
            for k in d:
                out.append(k)
            return out

        keys = [1, 2, 3]
        vals = [0.1, 0.2, 0.3]

        self.assertEqual(
            foo(keys, vals),
            [1, 2, 3]
        )

    def test_dict_contains(self):
        """
        Exercise operator.contains
        """
        @njit
        def foo(keys, vals, checklist):
            d = dictobject.new_dict(int32, float64)
            # insertion
            for k, v in zip(keys, vals):
                d[k] = v
            out = []
            for k in checklist:
                out.append(k in d)
            return out

        keys = [1, 2, 3]
        vals = [0.1, 0.2, 0.3]

        self.assertEqual(
            foo(keys, vals, [2, 3, 4, 1, 0]),
            [True, True, False, True, False],
        )

    def test_dict_copy(self):
        """
        Exercise dict.copy
        """
        @njit
        def foo(keys, vals):
            d = dictobject.new_dict(int32, float64)
            # insertion
            for k, v in zip(keys, vals):
                d[k] = v
            return list(d.copy().items())

        keys = list(range(20))
        vals = [x + i / 100 for i, x in enumerate(keys)]
        out = foo(keys, vals)
        self.assertEqual(out, list(zip(keys, vals)))

    def test_dict_setdefault(self):
        """
        Exercise dict.setdefault
        """
        @njit
        def foo():
            d = dictobject.new_dict(int32, float64)
            d.setdefault(1, 1.2) # used because key is not in
            a = d.get(1)
            d[1] = 2.3
            b = d.get(1)
            d[2] = 3.4
            d.setdefault(2, 4.5)  # not used because key is in
            c = d.get(2)
            return a, b, c

        self.assertEqual(foo(), (1.2, 2.3, 3.4))

    def test_dict_equality(self):
        """
        Exercise dict.__eq__ and .__ne__
        """
        @njit
        def foo(na, nb, fa, fb):
            da = dictobject.new_dict(int32, float64)
            db = dictobject.new_dict(int32, float64)
            for i in range(na):
                da[i] = i * fa
            for i in range(nb):
                db[i] = i * fb
            return da == db, da != db

        # Same keys and values
        self.assertEqual(foo(10, 10, 3, 3), (True, False))
        # Same keys and diff values
        self.assertEqual(foo(10, 10, 3, 3.1), (False, True))
        # LHS has more keys
        self.assertEqual(foo(11, 10, 3, 3), (False, True))
        # RHS has more keys
        self.assertEqual(foo(10, 11, 3, 3), (False, True))

    def test_dict_equality_more(self):
        """
        Exercise dict.__eq__
        """
        @njit
        def foo(ak, av, bk, bv):
            # The key-value types are different in the two dictionaries
            da = dictobject.new_dict(int32, float64)
            db = dictobject.new_dict(int64, float32)
            for i in range(len(ak)):
                da[ak[i]] = av[i]
            for i in range(len(bk)):
                db[bk[i]] = bv[i]
            return da == db

        # Simple equal case
        ak = [1, 2, 3]
        av = [2, 3, 4]
        bk = [1, 2, 3]
        bv = [2, 3, 4]
        self.assertTrue(foo(ak, av, bk, bv))

        # Equal with replacement
        ak = [1, 2, 3]
        av = [2, 3, 4]
        bk = [1, 2, 2, 3]
        bv = [2, 1, 3, 4]
        self.assertTrue(foo(ak, av, bk, bv))

        # Diff values
        ak = [1, 2, 3]
        av = [2, 3, 4]
        bk = [1, 2, 3]
        bv = [2, 1, 4]
        self.assertFalse(foo(ak, av, bk, bv))

        # Diff keys
        ak = [0, 2, 3]
        av = [2, 3, 4]
        bk = [1, 2, 3]
        bv = [2, 3, 4]
        self.assertFalse(foo(ak, av, bk, bv))

    def test_dict_equality_diff_type(self):
        """
        Exercise dict.__eq__
        """
        @njit
        def foo(na, b):
            da = dictobject.new_dict(int32, float64)
            for i in range(na):
                da[i] = i
            return da == b

        # dict != int
        self.assertFalse(foo(10, 1))
        # dict != tuple[int]
        self.assertFalse(foo(10, (1,)))

    def test_dict_box(self):
        """
        Exercise dictobject._box
        """
        @njit
        def make_content(nelem):
            for i in range(nelem):
                yield i, i + (i + 1) / 100

        @njit
        def boxer(nelem):
            d = dictobject.new_dict(int32, float64)
            for k, v in make_content(nelem):
                d[k] = v
            return dictobject._box(d)

        dcttype = types.DictType(int32, float64)

        @njit
        def unboxer(mi):
            d = dictobject._unbox(mi, dcttype)
            return list(d.items())

        mi = boxer(10)
        self.assertEqual(mi.refcount, 1)

        got = unboxer(mi)
        expected = list(make_content.py_func(10))
        self.assertEqual(got, expected)


class TestDictPy(MemoryLeakMixin, TestCase):
    def test_basic(self):
        d = NBDict.empty(int32, float32)
        # len
        self.assertEqual(len(d), 0)
        # setitems
        d[1] = 1
        d[2] = 2.3
        d[3] = 3.4
        self.assertEqual(len(d), 3)
        # keys
        self.assertEqual(list(d.keys()), [1, 2, 3])
        # values
        for x, y in zip(list(d.values()), [1, 2.3, 3.4]):
            self.assertAlmostEqual(x, y, places=4)
        # getitem
        self.assertAlmostEqual(d[1], 1)
        self.assertAlmostEqual(d[2], 2.3, places=4)
        self.assertAlmostEqual(d[3], 3.4, places=4)
        # deltiem
        del d[2]
        self.assertEqual(len(d), 2)
        # get
        self.assertIsNone(d.get(2))
        # setdefault
        d.setdefault(2, 100)
        d.setdefault(3, 200)
        self.assertEqual(d[2], 100)
        self.assertAlmostEqual(d[3], 3.4, places=4)

    def test_copy_from_dict(self):
        expect = {k: float(v) for k, v in zip(range(10), range(10, 20))}
        nbd = NBDict.empty(int32, float64)
        for k, v in expect.items():
            nbd[k] = v
        got = dict(nbd)
        self.assertEqual(got, expect)

    def test_compiled(self):
        @njit
        def producer():
            d = NBDict.empty(int32, float64)
            d[1] = 1.23
            return d

        @njit
        def consumer(d):
            return d[1]

        d = producer()
        val = consumer(d)
        self.assertEqual(val, 1.23)
