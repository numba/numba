from __future__ import print_function, division, absolute_import

from numba import njit, types
from numba import unittest_support as unittest
from numba.extending import overload
from functools import reduce


class TestMap(unittest.TestCase):

    def test_basic_map_external_func(self):
        func = njit(lambda x: x + 10)

        def impl():
            return [y for y in map(func, range(10))]

        cfunc = njit(impl)

        self.assertEqual(impl(), cfunc())

    def test_basic_map_closure(self):
        def impl():
            return [y for y in map(lambda x: x + 10, range(10))]

        cfunc = njit(impl)

        self.assertEqual(impl(), cfunc())

    def test_basic_map_closure_multiple_iterator(self):
        def impl():
            args = range(10), range(10, 20)
            return [y for y in map(lambda a, b: (a + 10, b + 5), *args)]

        cfunc = njit(impl)

        self.assertEqual(impl(), cfunc())


class TestFilter(unittest.TestCase):

    def test_basic_filter_external_func(self):
        func = njit(lambda x: x > 0)

        def impl():
            return [y for y in filter(func, range(-10, 10))]

        cfunc = njit(impl)

        self.assertEqual(impl(), cfunc())

    def test_basic_filter_closure(self):
        def impl():
            return [y for y in filter(lambda x: x > 0, range(-10, 10))]

        cfunc = njit(impl)

        self.assertEqual(impl(), cfunc())

    def test_basic_filter_none_func(self):
        def impl():
            return [y for y in filter(None, range(-10, 10))]

        cfunc = njit(impl)

        self.assertEqual(impl(), cfunc())


class TestReduce(unittest.TestCase):

    def test_basic_reduce_external_func(self):
        func = njit(lambda x, y: x + y)

        def impl():
            return reduce(func, range(-10, 10))

        cfunc = njit(impl)

        self.assertEqual(impl(), cfunc())

    def test_basic_reduce_closure(self):

        def impl():
            def func(x, y):
                return x + y
            return reduce(func, range(-10, 10), 100)

        cfunc = njit(impl)

        self.assertEqual(impl(), cfunc())


class TestSpecialMap(unittest.TestCase):
    def test_map_tuple_basic(self):
        from numba.special import map_tuple

        def gen(decor=lambda x: x):
            @decor
            def foo(a, b, c):
                tup = (c, b, a, b, a, 1.2)
                return bar(tup)

            @decor
            def bar(tup):
                f = lambda x: x + x
                return map_tuple(f, tup)

            return foo(12, "b", 3j)

        self.assertEqual(gen(), gen(njit))

    def test_map_tuple_overload(self):
        from numba.special import map_tuple

        def column_op(seq):
            pass

        @overload(column_op)
        def _column_op(seq):
            if isinstance(seq.dtype, types.Integer):
                return lambda seq: [x * 2 for x in seq]
            elif seq.dtype == types.unicode_type:
                return lambda seq: [x.strip() for x in seq]

        @njit
        def complex_example():
            columns = (list(range(10)), ['apple', 'orange', 'mango  '])
            output1 = map_tuple(len, columns)
            output2 = map_tuple(column_op, columns)
            return output1, output2

        output1, output2 = complex_example()
        self.assertEqual(output1, (10, 3))
        self.assertEqual(output2, (
            [x * 2 for x in range(10)],
            ['apple', 'orange', 'mango'],
        ))


if __name__ == '__main__':
    unittest.main()
