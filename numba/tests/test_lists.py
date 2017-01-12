from __future__ import print_function

from collections import namedtuple
import contextlib
import itertools
import math
import sys

from numba.compiler import compile_isolated, Flags
from numba import jit, types
import numba.unittest_support as unittest
from numba import testing
from .support import TestCase, MemoryLeakMixin, tag


enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

force_pyobj_flags = Flags()
force_pyobj_flags.set("force_pyobject")

Point = namedtuple('Point', ('a', 'b'))


def noop(x):
    pass

def unbox_usecase(x):
    """
    Expect a list of numbers
    """
    res = 0
    for v in x:
        res += v
    return res

def unbox_usecase2(x):
    """
    Expect a list of tuples
    """
    res = 0
    for v in x:
        res += len(v)
    return res

def unbox_usecase3(x):
    """
    Expect a (number, list of numbers) tuple.
    """
    a, b = x
    res = a
    for v in b:
        res += v
    return res

def unbox_usecase4(x):
    """
    Expect a (number, list of tuples) tuple.
    """
    a, b = x
    res = a
    for v in b:
        res += len(v)
    return res


def create_list(x, y, z):
    return [x, y, z]

def create_nested_list(x, y, z, a, b, c):
    return [[x, y, z], [a, b, c]]

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

def list_extend(n):
    l = []
    # A non-list iterable and a list
    l.extend(range(n))
    l.extend(l[:-1])
    l.extend(range(n, 0, -1))
    return l

def list_extend_heterogenous(n):
    l = []
    # Extend with various iterables, including lists, with different types
    l.extend(range(n))
    l.extend(l[:-1])
    l.extend((5, 42))
    l.extend([123.0])
    return l

def list_pop0(n):
    l = list(range(n))
    res = 0
    while len(l) > 0:
        res += len(l) * l.pop()
    return res

def list_pop1(n, i):
    l = list(range(n))
    x = l.pop(i)
    return x, l

def list_len(n):
    l = list(range(n))
    return len(l)

def list_getitem(n):
    l = list(range(n))
    res = 0
    # Positive indices
    for i in range(len(l)):
        res += i * l[i]
    # Negative indices
    for i in range(-len(l), 0):
        res -= i * l[i]
    return res

def list_setitem(n):
    l = list(range(n))
    res = 0
    # Positive indices
    for i in range(len(l)):
        l[i] = i * l[i]
    # Negative indices
    for i in range(-len(l), 0):
        l[i] = i * l[i]
    for i in range(len(l)):
        res += l[i]
    return res

def list_getslice2(n, start, stop):
    l = list(range(n))
    return l[start:stop]

def list_getslice3(n, start, stop, step):
    l = list(range(n))
    return l[start:stop:step]

def list_setslice2(n, n_source, start, stop):
    # Generic setslice with size change
    l = list(range(n))
    v = list(range(100, 100 + n_source))
    l[start:stop] = v
    return l

def list_setslice3(n, start, stop, step):
    l = list(range(n))
    v = l[start:stop:step]
    for i in range(len(v)):
        v[i] += 100
    l[start:stop:step] = v
    return l

def list_setslice3_arbitrary(n, n_src, start, stop, step):
    l = list(range(n))
    l[start:stop:step] = list(range(100, 100 + n_src))
    return l

def list_delslice0(n):
    l = list(range(n))
    del l[:]
    return l

def list_delslice1(n, start, stop):
    l = list(range(n))
    del l[start:]
    del l[:stop]
    return l

def list_delslice2(n, start, stop):
    l = list(range(n))
    del l[start:stop]
    return l

def list_clear(n):
    l = list(range(n))
    l.clear()
    return l

def list_copy(n):
    l = list(range(n))
    ll = l.copy()
    l.append(42)
    return l, ll

def list_iteration(n):
    l = list(range(n))
    res = 0
    for i, v in enumerate(l):
        res += i * v
    return res

def list_contains(n):
    l = list(range(n))
    return (0 in l, 1 in l, n - 1 in l, n in l,
            0 not in l, 1 not in l, n - 1 not in l, n not in l,
            )

def list_index1(n, v):
    l = list(range(n, 0, -1))
    return l.index(v)

def list_index2(n, v, start):
    l = list(range(n, 0, -1))
    return l.index(v, start)

def list_index3(n, v, start, stop):
    l = list(range(n, 0, -1))
    return l.index(v, start, stop)

def list_remove(n, v):
    l = list(range(n - 1, -1, -1))
    l.remove(v)
    return l

def list_insert(n, pos, v):
    l = list(range(0, n))
    l.insert(pos, v)
    return l

def list_count(n, v):
    l = []
    for x in range(n):
        l.append(x & 3)
    return l.count(v)

def list_reverse(n):
    l = list(range(n))
    l.reverse()
    return l

def list_add(m, n):
    a = list(range(0, m))
    b = list(range(100, 100 + n))
    res = a + b
    res.append(42)   # check result is a copy
    return a, b, res

def list_add_heterogenous():
    a = [1]
    b = [2.0]
    c = a + b
    d = b + a
    # check result is a copy
    a.append(3)
    b.append(4.0)
    return a, b, c, d

def list_add_inplace(m, n):
    a = list(range(0, m))
    b = list(range(100, 100 + n))
    a += b
    return a, b

def list_add_inplace_heterogenous():
    a = [1]
    b = [2.0]
    a += b
    b += a
    return a, b

def list_mul(n, v):
    a = list(range(n))
    return a * v

def list_mul_inplace(n, v):
    a = list(range(n))
    a *= v
    return a

def list_bool(n):
    a = list(range(n))
    return bool(a), (True if a else False)

def eq_usecase(a, b):
    return list(a) == list(b)

def ne_usecase(a, b):
    return list(a) != list(b)

def gt_usecase(a, b):
    return list(a) > list(b)

def ge_usecase(a, b):
    return list(a) >= list(b)

def lt_usecase(a, b):
    return list(a) < list(b)

def le_usecase(a, b):
    return list(a) <= list(b)

def identity_usecase(n):
    a = list(range(n))
    b = a
    c = a[:]
    return (a is b), (a is not b), (a is c), (a is not c)

def bool_list_usecase():
    # Exercise getitem, setitem, iteration with bool values (issue #1373)
    l = [False]
    l[0] = True
    x = False
    for v in l:
        x = x ^ v
    return l, x

def reflect_simple(l, ll):
    x = l.pop()
    y = l.pop()
    l[0] = 42.
    l.extend(ll)
    return l, x, y

def reflect_conditional(l, ll):
    # `l` may or may not actually reflect a Python list
    if ll[0]:
        l = [11., 22., 33., 44.]
    x = l.pop()
    y = l.pop()
    l[0] = 42.
    l.extend(ll)
    return l, x, y

def reflect_exception(l):
    l.append(42)
    raise ZeroDivisionError

def reflect_dual(l, ll):
    l.append(ll.pop())
    return l is ll


class TestLists(MemoryLeakMixin, TestCase):

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
        # Use various sizes, to stress the allocation algorithm
        for n in [0, 3, 16, 70, 400]:
            eq = self.assertPreciseEqual if precise else self.assertEqual
            eq(cfunc(n), pyfunc(n))

    def test_constructor(self):
        self.check_unary_with_size(list_constructor)

    def test_append(self):
        self.check_unary_with_size(list_append)

    @tag('important')
    def test_append_heterogenous(self):
        self.check_unary_with_size(list_append_heterogenous, precise=False)

    def test_extend(self):
        self.check_unary_with_size(list_extend)

    @tag('important')
    def test_extend_heterogenous(self):
        self.check_unary_with_size(list_extend_heterogenous, precise=False)

    def test_pop0(self):
        self.check_unary_with_size(list_pop0)

    @tag('important')
    def test_pop1(self):
        pyfunc = list_pop1
        cfunc = jit(nopython=True)(pyfunc)
        for n in [5, 40]:
            for i in [0, 1, n - 2, n - 1, -1, -2, -n + 3, -n + 1]:
                expected = pyfunc(n, i)
                self.assertPreciseEqual(cfunc(n, i), expected)

    def test_pop_errors(self):
        # XXX References are leaked when an exception is raised
        self.disable_leak_check()
        cfunc = jit(nopython=True)(list_pop1)
        with self.assertRaises(IndexError) as cm:
            cfunc(0, 5)
        self.assertEqual(str(cm.exception), "pop from empty list")
        with self.assertRaises(IndexError) as cm:
            cfunc(1, 5)
        self.assertEqual(str(cm.exception), "pop index out of range")

    def test_insert(self):
        pyfunc = list_insert
        cfunc = jit(nopython=True)(pyfunc)
        for n in [5, 40]:
            indices = [0, 1, n - 2, n - 1, n + 1, -1, -2, -n + 3, -n - 1] 
            for i in indices:
                expected = pyfunc(n, i, 42)
                self.assertPreciseEqual(cfunc(n, i, 42), expected)

    def test_len(self):
        self.check_unary_with_size(list_len)

    @tag('important')
    def test_getitem(self):
        self.check_unary_with_size(list_getitem)

    @tag('important')
    def test_setitem(self):
        self.check_unary_with_size(list_setitem)

    def check_slicing2(self, pyfunc):
        cfunc = jit(nopython=True)(pyfunc)
        sizes = [5, 40]
        for n in sizes:
            indices = [0, 1, n - 2, -1, -2, -n + 3, -n - 1, -n]
            for start, stop in itertools.product(indices, indices):
                expected = pyfunc(n, start, stop)
                self.assertPreciseEqual(cfunc(n, start, stop), expected)

    def test_getslice2(self):
        self.check_slicing2(list_getslice2)

    def test_setslice2(self):
        pyfunc = list_setslice2
        cfunc = jit(nopython=True)(pyfunc)
        sizes = [5, 40]
        for n, n_src in itertools.product(sizes, sizes):
            indices = [0, 1, n - 2, -1, -2, -n + 3, -n - 1, -n]
            for start, stop in itertools.product(indices, indices):
                expected = pyfunc(n, n_src, start, stop)
                self.assertPreciseEqual(cfunc(n, n_src, start, stop), expected)

    @tag('important')
    def test_getslice3(self):
        pyfunc = list_getslice3
        cfunc = jit(nopython=True)(pyfunc)
        for n in [10]:
            indices = [0, 1, n - 2, -1, -2, -n + 3, -n - 1, -n]
            steps = [4, 1, -1, 2, -3]
            for start, stop, step in itertools.product(indices, indices, steps):
                expected = pyfunc(n, start, stop, step)
                self.assertPreciseEqual(cfunc(n, start, stop, step), expected)

    @tag('important')
    def test_setslice3(self):
        pyfunc = list_setslice3
        cfunc = jit(nopython=True)(pyfunc)
        for n in [10]:
            indices = [0, 1, n - 2, -1, -2, -n + 3, -n - 1, -n]
            steps = [4, 1, -1, 2, -3]
            for start, stop, step in itertools.product(indices, indices, steps):
                expected = pyfunc(n, start, stop, step)
                self.assertPreciseEqual(cfunc(n, start, stop, step), expected)

    def test_setslice3_resize(self):
        # XXX References are leaked when an exception is raised
        self.disable_leak_check()
        pyfunc = list_setslice3_arbitrary
        cfunc = jit(nopython=True)(pyfunc)
        # step == 1 => can resize
        cfunc(5, 10, 0, 2, 1)
        # step != 1 => cannot resize
        with self.assertRaises(ValueError) as cm:
            cfunc(5, 100, 0, 3, 2)
        self.assertIn("cannot resize", str(cm.exception))

    def test_delslice0(self):
        self.check_unary_with_size(list_delslice0)

    def test_delslice1(self):
        self.check_slicing2(list_delslice1)

    @tag('important')
    def test_delslice2(self):
        self.check_slicing2(list_delslice2)

    def test_invalid_slice(self):
        self.disable_leak_check()
        pyfunc = list_getslice3
        cfunc = jit(nopython=True)(pyfunc)
        with self.assertRaises(ValueError) as cm:
            cfunc(10, 1, 2, 0)
        self.assertEqual(str(cm.exception), "slice step cannot be zero")

    def test_iteration(self):
        self.check_unary_with_size(list_iteration)

    @tag('important')
    def test_reverse(self):
        self.check_unary_with_size(list_reverse)

    def test_contains(self):
        self.check_unary_with_size(list_contains)

    def check_index_result(self, pyfunc, cfunc, args):
        try:
            expected = pyfunc(*args)
        except ValueError:
            with self.assertRaises(ValueError):
                cfunc(*args)
        else:
            self.assertPreciseEqual(cfunc(*args), expected)

    def test_index1(self):
        self.disable_leak_check()
        pyfunc = list_index1
        cfunc = jit(nopython=True)(pyfunc)
        for v in (0, 1, 5, 10, 99999999):
            self.check_index_result(pyfunc, cfunc, (16, v))

    def test_index2(self):
        self.disable_leak_check()
        pyfunc = list_index2
        cfunc = jit(nopython=True)(pyfunc)
        n = 16
        for v in (0, 1, 5, 10, 99999999):
            indices = [0, 1, n - 2, n - 1, n + 1, -1, -2, -n + 3, -n - 1]
            for start in indices:
                self.check_index_result(pyfunc, cfunc, (16, v, start))

    def test_index3(self):
        self.disable_leak_check()
        pyfunc = list_index3
        cfunc = jit(nopython=True)(pyfunc)
        n = 16
        for v in (0, 1, 5, 10, 99999999):
            indices = [0, 1, n - 2, n - 1, n + 1, -1, -2, -n + 3, -n - 1]
            for start, stop in itertools.product(indices, indices):
                self.check_index_result(pyfunc, cfunc, (16, v, start, stop))

    def test_remove(self):
        pyfunc = list_remove
        cfunc = jit(nopython=True)(pyfunc)
        n = 16
        for v in (0, 1, 5, 15):
            expected = pyfunc(n, v)
            self.assertPreciseEqual(cfunc(n, v), expected)

    def test_remove_error(self):
        self.disable_leak_check()
        pyfunc = list_remove
        cfunc = jit(nopython=True)(pyfunc)
        with self.assertRaises(ValueError) as cm:
            cfunc(10, 42)
        self.assertEqual(str(cm.exception), "list.remove(x): x not in list")

    def test_count(self):
        pyfunc = list_count
        cfunc = jit(nopython=True)(pyfunc)
        for v in range(5):
            self.assertPreciseEqual(cfunc(18, v), pyfunc(18, v))

    @unittest.skipUnless(sys.version_info >= (3, 3),
                         "list.clear() needs Python 3.3+")
    def test_clear(self):
        self.check_unary_with_size(list_clear)

    @unittest.skipUnless(sys.version_info >= (3, 3),
                         "list.copy() needs Python 3.3+")
    def test_copy(self):
        self.check_unary_with_size(list_copy)

    def check_add(self, pyfunc):
        cfunc = jit(nopython=True)(pyfunc)
        sizes = [0, 3, 50, 300]
        for m, n in itertools.product(sizes, sizes):
            expected = pyfunc(m, n)
            self.assertPreciseEqual(cfunc(m, n), expected)

    def test_add(self):
        self.check_add(list_add)

    def test_add_heterogenous(self):
        pyfunc = list_add_heterogenous
        cfunc = jit(nopython=True)(pyfunc)
        expected = pyfunc()
        self.assertEqual(cfunc(), expected)

    def test_add_inplace(self):
        self.check_add(list_add_inplace)

    def test_add_inplace_heterogenous(self):
        pyfunc = list_add_inplace_heterogenous
        cfunc = jit(nopython=True)(pyfunc)
        expected = pyfunc()
        self.assertEqual(cfunc(), expected)

    def check_mul(self, pyfunc):
        cfunc = jit(nopython=True)(pyfunc)
        for n in [0, 3, 50, 300]:
            for v in [1, 2, 3, 0, -1, -42]:
                expected = pyfunc(n, v)
                self.assertPreciseEqual(cfunc(n, v), expected)

    def test_mul(self):
        self.check_mul(list_mul)

    def test_mul_inplace(self):
        self.check_mul(list_mul_inplace)

    @unittest.skipUnless(sys.maxsize >= 2**32,
                         "need a 64-bit system to test for MemoryError")
    def test_mul_error(self):
        self.disable_leak_check()
        pyfunc = list_mul
        cfunc = jit(nopython=True)(pyfunc)
        # Fail in malloc()
        with self.assertRaises(MemoryError):
            cfunc(1, 2**58)
        # Overflow size computation when multiplying by item size
        with self.assertRaises(MemoryError):
            cfunc(1, 2**62)

    def test_bool(self):
        pyfunc = list_bool
        cfunc = jit(nopython=True)(pyfunc)
        for n in [0, 1, 3]:
            expected = pyfunc(n)
            self.assertPreciseEqual(cfunc(n), expected)

    def test_list_passing(self):
        # Check one can pass a list from a Numba function to another
        @jit(nopython=True)
        def inner(lst):
            return len(lst), lst[-1]

        @jit(nopython=True)
        def outer(n):
            l = list(range(n))
            return inner(l)

        self.assertPreciseEqual(outer(5), (5, 4))

    def _test_compare(self, pyfunc):
        def eq(args):
            self.assertIs(cfunc(*args), pyfunc(*args),
                          "mismatch for arguments %s" % (args,))

        cfunc = jit(nopython=True)(pyfunc)
        eq(((1, 2), (1, 2)))
        eq(((1, 2, 3), (1, 2)))
        eq(((1, 2), (1, 2, 3)))
        eq(((1, 2, 4), (1, 2, 3)))
        eq(((1.0, 2.0, 3.0), (1, 2, 3)))
        eq(((1.0, 2.0, 3.5), (1, 2, 3)))

    def test_eq(self):
        self._test_compare(eq_usecase)

    def test_ne(self):
        self._test_compare(ne_usecase)

    def test_le(self):
        self._test_compare(le_usecase)

    def test_lt(self):
        self._test_compare(lt_usecase)

    def test_ge(self):
        self._test_compare(ge_usecase)

    def test_gt(self):
        self._test_compare(gt_usecase)

    def test_identity(self):
        pyfunc = identity_usecase
        cfunc = jit(nopython=True)(pyfunc)
        self.assertPreciseEqual(cfunc(3), pyfunc(3))

    def test_bool_list(self):
        # Check lists of bools compile and run successfully
        pyfunc = bool_list_usecase
        cfunc = jit(nopython=True)(pyfunc)
        self.assertPreciseEqual(cfunc(), pyfunc())


class TestUnboxing(MemoryLeakMixin, TestCase):
    """
    Test unboxing of Python lists into native Numba lists.
    """

    @contextlib.contextmanager
    def assert_type_error(self, msg):
        with self.assertRaises(TypeError) as raises:
            yield
        if msg is not None:
            self.assertRegexpMatches(str(raises.exception), msg)

    def check_unary(self, pyfunc):
        cfunc = jit(nopython=True)(pyfunc)
        def check(arg):
            expected = pyfunc(arg)
            got = cfunc(arg)
            self.assertPreciseEqual(got, expected)
        return check

    def test_numbers(self):
        check = self.check_unary(unbox_usecase)
        check([1, 2])
        check([1j, 2.5j])

    def test_tuples(self):
        check = self.check_unary(unbox_usecase2)
        check([(1, 2), (3, 4)])
        check([(1, 2j), (3, 4j)])
        check([(), (), ()])

    @tag('important')
    def test_list_inside_tuple(self):
        check = self.check_unary(unbox_usecase3)
        check((1, [2, 3, 4]))

    def test_list_of_tuples_inside_tuple(self):
        check = self.check_unary(unbox_usecase4)
        check((1, [(2,), (3,)]))

    def test_errors(self):
        # See #1545 and #1594: error checking should ensure the list is
        # homogenous
        msg = "can't unbox heterogenous list"
        pyfunc = noop
        cfunc = jit(nopython=True)(pyfunc)
        lst = [1, 2.5]
        with self.assert_type_error(msg):
            cfunc(lst)
        # The list hasn't been changed (bogus reflecting)
        self.assertEqual(lst, [1, 2.5])
        with self.assert_type_error(msg):
            cfunc([1, 2j])
        # Same when the list is nested in a tuple or namedtuple
        with self.assert_type_error(msg):
            cfunc((1, [1, 2j]))
        with self.assert_type_error(msg):
            cfunc(Point(1, [1, 2j]))
        # Issue #1638: tuples of different size.
        # Note the check is really on the tuple side.
        lst = [(1,), (2, 3)]
        with self.assertRaises(ValueError) as raises:
            cfunc(lst)
        self.assertEqual(str(raises.exception),
                         "size mismatch for tuple, expected 1 element(s) but got 2")


class TestListReflection(MemoryLeakMixin, TestCase):
    """
    Test reflection of native Numba lists on Python list objects.
    """

    def check_reflection(self, pyfunc):
        cfunc = jit(nopython=True)(pyfunc)
        samples = [([1., 2., 3., 4.], [0.]),
                   ([1., 2., 3., 4.], [5., 6., 7., 8., 9.]),
                   ]
        for dest, src in samples:
            expected = list(dest)
            got = list(dest)
            pyres = pyfunc(expected, src)
            with self.assertRefCount(got, src):
                cres = cfunc(got, src)
                self.assertPreciseEqual(cres, pyres)
                self.assertPreciseEqual(expected, got)
                self.assertEqual(pyres[0] is expected, cres[0] is got)
                del pyres, cres

    def test_reflect_simple(self):
        self.check_reflection(reflect_simple)

    def test_reflect_conditional(self):
        self.check_reflection(reflect_conditional)

    def test_reflect_exception(self):
        """
        When the function exits with an exception, lists should still be
        reflected.
        """
        pyfunc = reflect_exception
        cfunc = jit(nopython=True)(pyfunc)
        l = [1, 2, 3]
        with self.assertRefCount(l):
            with self.assertRaises(ZeroDivisionError):
                cfunc(l)
            self.assertPreciseEqual(l, [1, 2, 3, 42])

    @tag('important')
    def test_reflect_same_list(self):
        """
        When the same list object is reflected twice, behaviour should
        be consistent.
        """
        pyfunc = reflect_dual
        cfunc = jit(nopython=True)(pyfunc)
        pylist = [1, 2, 3]
        clist = pylist[:]
        expected = pyfunc(pylist, pylist)
        got = cfunc(clist, clist)
        self.assertPreciseEqual(expected, got)
        self.assertPreciseEqual(pylist, clist)
        self.assertPreciseEqual(sys.getrefcount(pylist), sys.getrefcount(clist))

    def test_reflect_clean(self):
        """
        When the list wasn't mutated, no reflection should take place.
        """
        cfunc = jit(nopython=True)(noop)
        # Use a complex, as Python integers can be cached
        l = [12.5j]
        ids = [id(x) for x in l]
        cfunc(l)
        self.assertEqual([id(x) for x in l], ids)


if __name__ == '__main__':
    unittest.main()
