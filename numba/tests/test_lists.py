
from __future__ import print_function

import math

from numba.compiler import compile_isolated, Flags
from numba import jit, types
import numba.unittest_support as unittest
from numba import testing
from .support import TestCase, MemoryLeakMixin


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

def list_comprehension1():
    return sum([x**2 for x in range(10)])

def list_comprehension2():
    return sum([x for x in range(10) if x % 2 == 0])

def list_comprehension3():
    return sum([math.pow(x, 2) for x in range(10)])

def list_comprehension4():
    return sum([x * y for x in range(10) for y in range(10)])

def list_comprehension5():
    return [x * 2 for x in range(10)]

def list_comprehension6():
    return [[x for x in range(y)] for y in range(3)]


def list_constructor(n):
    return list(range(n))

def list_append(n):
    l = []
    l.append(42)
    for i in range(n):
        l.append(i)
    return l

def list_append_heterogenous(n):
    l = []
    l.append(42.0)
    for i in range(n):
        l.append(i)
    return l

def list_pop(n):
    l = list(range(n))
    res = 0
    while len(l) > 0:
        res += len(l) * l.pop()
    return res

def list_len(n):
    l = list(range(n))
    return len(l)


def list_extend(l1, l2):
    l1.extend(l2)
    return l1

def list_insert(l, i, x):
    l.insert(i, x)
    return l

def list_remove(l, x):
    l.remove(x)
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


class TestLists(MemoryLeakMixin, TestCase):

    def test_identity_func(self):
        pyfunc = identity_func
        with self.assertTypingError():
            cr = compile_isolated(pyfunc, (types.Dummy('list'),))
            cfunc = cr.entry_point
            l = range(10)
            self.assertEqual(cfunc(l), pyfunc(l))

    def test_create_list(self):
        pyfunc = create_list
        cr = compile_isolated(pyfunc, (types.int32, types.int32, types.int32))
        cfunc = cr.entry_point
        self.assertEqual(cfunc(1, 2, 3), pyfunc(1, 2, 3))

    def test_create_nested_list(self):
        pyfunc = create_nested_list
        with self.assertTypingError():
            cr = compile_isolated(pyfunc, (types.int32, types.int32, types.int32,
                types.int32, types.int32, types.int32))
            cfunc = cr.entry_point
            self.assertEqual(cfunc(1, 2, 3, 4, 5, 6), pyfunc(1, 2, 3, 4, 5, 6))

    @testing.allow_interpreter_mode
    def test_list_comprehension(self):
        list_tests = [list_comprehension1,
                      list_comprehension2,
                      list_comprehension3,
                      list_comprehension4,
                      list_comprehension5,
                      list_comprehension6]

        for test in list_tests:
            pyfunc = test
            cr = compile_isolated(pyfunc, ())
            cfunc = cr.entry_point
            self.assertEqual(cfunc(), pyfunc())

    def check_unary_with_size(self, pyfunc, precise=True):
        cfunc = jit(nopython=True)(pyfunc)
        # Exercises various sizes, for the allocation
        for n in [0, 2, 5, 16, 70, 400]:
            eq = self.assertPreciseEqual if precise else self.assertEqual
            eq(cfunc(n), pyfunc(n))

    def test_constructor(self):
        self.check_unary_with_size(list_constructor)

    def test_append(self):
        self.check_unary_with_size(list_append)

    def test_append_heterogenous(self):
        self.check_unary_with_size(list_append_heterogenous, precise=False)

    def test_pop(self):
        self.check_unary_with_size(list_pop)

    def test_len(self):
        self.check_unary_with_size(list_len)


if __name__ == '__main__':
    unittest.main()

