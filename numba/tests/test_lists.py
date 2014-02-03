
from __future__ import print_function
import numba.unittest_support as unittest
from numba.compiler import compile_isolated, Flags
from numba import types, utils
from numba.tests import usecases
import math
import numpy as np

enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

force_pyobj_flags = Flags()
force_pyobj_flags.set("force_pyobject")


def identity_func(l):
    return l

def create_list(x, y, z):
    return [x, y, z]

def create_nested_list(x, y, z, a, b, c):
    return [[x, y, z], [a, b, c]]

def get_list_item(l, i):
    return l[i]

def get_list_slice(l, start, stop, step):
    return l[start:stop:step]

def set_list_item(l, i, x):
    l[i] = x
    return l

def set_list_slice(l, start, stop, step, x):
    l[start:stop:step] = x
    return l

def get_list_len(l):
    return len(l)

def list_comprehension1():
    return sum([x**2 for x in range(10)])

def list_comprehension2():
    return sum([x for x in range(10) if x % 2 == 0])

def list_comprehension3():
    return sum([math.pow(x, 2) for x in range(10)])

def list_comprehension4():
    return sum([x * y for x in range(10) for y in range(10)])

def list_append(l, x):
    l.append(x)
    return l

def list_extend(l1, l2):
    l1.extend(l2)
    return l1

def list_insert(l, i, x):
    l.insert(i, x)
    return l

def list_remove(l, x):
    l.remove(x)
    return l

def list_pop(l):
    l.pop()
    return l

def list_index(l, x):
    return l.index(x)

def list_count(l, x):
    return l.count(x)

def list_sort(l):
    l.sort()
    return l

def list_reverse(l):
    l.reverse()
    return l


class TestLists(unittest.TestCase):
    @unittest.expectedFailure
    def test_identity_func(self):
        pyfunc = identity_func
        cr = compile_isolated(pyfunc, (types.Dummy('list'),))
        cfunc = cr.entry_point
        l = range(10)
        self.assertEqual(cfunc(l), pyfunc(l))

    @unittest.expectedFailure
    def test_create_list(self):
        pyfunc = create_list
        cr = compile_isolated(pyfunc, (types.int32, types.int32, types.int32))
        cfunc = cr.entry_point
        self.assertEqual(cfunc(1, 2, 3), pyfunc(1, 2, 3))

    @unittest.expectedFailure
    def test_create_nested_list(self):
        pyfunc = create_nested_list
        cr = compile_isolated(pyfunc, (types.int32, types.int32, types.int32,
            types.int32, types.int32, types.int32))
        cfunc = cr.entry_point
        self.assertEqual(cfunc(1, 2, 3, 4, 5, 6), pyfunc(1, 2, 3, 4, 5, 6))

    @unittest.expectedFailure
    def test_get_list_item(self):
        pyfunc = get_list_item
        cr = compile_isolated(pyfunc, (types.int32, types.int32, types.int32))
        cfunc = cr.entry_point
        self.assertEqual(cfunc(1,2,3), pyfunc(1,2,3))

    @unittest.expectedFailure
    def test_get_list_slice(self):
        pyfunc = get_list_slice
        cr = compile_isolated(pyfunc, (types.Dummy('list'),
            types.int32, types.int32, types.int32))
        cfunc = cr.entry_point
        l = range(10)
        self.assertEqual(cfunc(l, 0, 10, 2), pyfunc(l, 0, 10, 2))

    @unittest.expectedFailure
    def test_set_list_item(self):
        pyfunc = set_list_item
        cr = compile_isolated(pyfunc, (types.Dummy('list'),
            types.int32, types.int32))
        cfunc = cr.entry_point
        l = range(10)
        self.assertEqual(cfunc(l, 0, 999), pyfunc(l, 0, 999))

    @unittest.expectedFailure
    def test_set_list_slice(self):
        pyfunc = set_list_slice
        cr = compile_isolated(pyfunc, (types.Dummy('list'),
            types.int32, types.int32, types.int32, types.int32))
        cfunc = cr.entry_point
        l = range(10)
        x = [999, 999, 999, 999, 999]
        self.assertEqual(cfunc(l, 0, 10, 2, x), pyfunc(l, 0, 10, 2, x))

    @unittest.expectedFailure
    def test_get_list_len(self):
        pyfunc = get_list_len
        cr = compile_isolated(pyfunc, (types.Dummy('list'),))
        cfunc = cr.entry_point
        l = range(10)
        self.assertEqual(cfunc(l), pyfunc(l))

    @unittest.expectedFailure
    def test_list_comprehension(self):
        list_tests = [list_comprehension1,
                      list_comprehension2,
                      list_comprehension3,
                      list_comprehension3]

        for test in list_tests:
            pyfunc = test
            cr = compile_isolated(pyfunc, ())
            cfunc = cr.entry_point
            self.assertEqual(cfunc(), pyfunc())

    @unittest.expectedFailure
    def test_list_append(self):
        pyfunc = list_append
        cr = compile_isolated(pyfunc, (types.Dummy('list'), types.int32))
        cfunc = cr.entry_point
        l = range(10)
        self.assertEqual(cfunc(l, 10), pyfunc(l, 10))

    @unittest.expectedFailure
    def test_list_extend(self):
        pyfunc = list_extend
        cr = compile_isolated(pyfunc, (types.Dummy('list'),
            types.Dummy('list')))
        cfunc = cr.entry_point
        l1 = range(10)
        l2 = range(10)
        self.assertEqual(cfunc(l1, l2), pyfunc(l1, l2))

    @unittest.expectedFailure
    def test_list_insert(self):
        pyfunc = list_insert
        cr = compile_isolated(pyfunc, (types.Dummy('list'),
            types.int32, types.int32))
        cfunc = cr.entry_point
        l = range(10)
        self.assertEqual(cfunc(l, 0, 999), pyfunc(l, 0, 999))

    @unittest.expectedFailure
    def test_list_remove(self):
        pyfunc = list_remove
        cr = compile_isolated(pyfunc, (types.Dummy('list'), types.int32))
        cfunc = cr.entry_point
        l = range(10)
        self.assertEqual(cfunc(l, 1), pyfunc(l, 1))

    @unittest.expectedFailure
    def test_list_pop(self):
        pyfunc = list_pop
        cr = compile_isolated(pyfunc, (types.Dummy('list'),))
        cfunc = cr.entry_point
        l = range(10)
        self.assertEqual(cfunc(l), pyfunc(l))

    @unittest.expectedFailure
    def test_list_index(self):
        pyfunc = list_index
        cr = compile_isolated(pyfunc, (types.Dummy('list'), types.int32))
        cfunc = cr.entry_point
        l = range(10)
        self.assertEqual(cfunc(l, 1), pyfunc(l, 1))

    @unittest.expectedFailure
    def test_list_count(self):
        pyfunc = list_count
        cr = compile_isolated(pyfunc, (types.Dummy('list'), types.int32))
        cfunc = cr.entry_point
        l = [1,1,2,1]
        self.assertEqual(cfunc(l, 1), pyfunc(l, 1))

    @unittest.expectedFailure
    def test_list_sort(self):
        pyfunc = list_sort
        cr = compile_isolated(pyfunc, (types.Dummy('list'),))
        cfunc = cr.entry_point
        l = np.random.randint(10, size=10)
        self.assertEqual(cfunc(l), pyfunc(l))

    @unittest.expectedFailure
    def test_list_reverse(self):
        pyfunc = list_reverse
        cr = compile_isolated(pyfunc, (types.Dummy('list'),))
        cfunc = cr.entry_point
        l = range(10)
        self.assertEqual(cfunc(l), pyfunc(l))


if __name__ == '__main__':
    unittest.main(buffer=True)

