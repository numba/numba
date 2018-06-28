from __future__ import print_function

import numpy as np

import numba.unittest_support as unittest
from numba.compiler import compile_isolated, Flags
from numba import numpy_support, types
from .support import TestCase, tag

enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

force_pyobj_flags = Flags()
force_pyobj_flags.set("force_pyobject")

no_pyobj_flags = Flags()


def int_tuple_iter_usecase():
    res = 0
    for i in (1, 2, 99, 3):
        res += i
    return res

def float_tuple_iter_usecase():
    res = 0.0
    for i in (1.5, 2.0, 99.3, 3.4):
        res += i
    return res

def tuple_tuple_iter_usecase():
    # Recursively homogeneous tuple type
    res = 0.0
    for i in ((1.5, 2.0), (99.3, 3.4), (1.8, 2.5)):
        for j in i:
            res += j
        res = res * 2
    return res

def enumerate_nested_tuple_usecase():
    res = 0.0
    for i, j in enumerate(((1.5, 2.0), (99.3, 3.4), (1.8, 2.5))):
        for l in j:
            res += i * l
        res = res * 2
    return res

def nested_enumerate_usecase():
    res = 0.0
    for i, (j, k) in enumerate(enumerate(((1.5, 2.0), (99.3, 3.4), (1.8, 2.5)))):
        for l in k:
            res += i * j * l
        res = res * 2
    return res

def scalar_iter_usecase(iterable):
    res = 0.0
    for x in iterable:
        res += x
    return res

def record_iter_usecase(iterable):
    res = 0.0
    for x in iterable:
        res += x.a * x.b
    return res

def record_iter_mutate_usecase(iterable):
    for x in iterable:
        x.a = x.a + x.b


record_dtype = np.dtype([('a', np.float64),
                         ('b', np.int32),
                         ])


class IterationTest(TestCase):

    def run_nullary_func(self, pyfunc, flags):
        cr = compile_isolated(pyfunc, (), flags=flags)
        cfunc = cr.entry_point
        expected = pyfunc()
        self.assertPreciseEqual(cfunc(), expected)

    def test_int_tuple_iter(self, flags=force_pyobj_flags):
        self.run_nullary_func(int_tuple_iter_usecase, flags)

    @tag('important')
    def test_int_tuple_iter_npm(self):
        self.test_int_tuple_iter(flags=no_pyobj_flags)

    # Type inference on tuples used to be hardcoded for ints, check
    # that it works for other types.

    def test_float_tuple_iter(self, flags=force_pyobj_flags):
        self.run_nullary_func(float_tuple_iter_usecase, flags)

    def test_float_tuple_iter_npm(self):
        self.test_float_tuple_iter(flags=no_pyobj_flags)

    def test_tuple_tuple_iter(self, flags=force_pyobj_flags):
        self.run_nullary_func(tuple_tuple_iter_usecase, flags)

    @tag('important')
    def test_tuple_tuple_iter_npm(self):
        self.test_tuple_tuple_iter(flags=no_pyobj_flags)

    def test_enumerate_nested_tuple(self, flags=force_pyobj_flags):
        self.run_nullary_func(enumerate_nested_tuple_usecase, flags)

    @tag('important')
    def test_enumerate_nested_tuple_npm(self):
        self.test_enumerate_nested_tuple(flags=no_pyobj_flags)

    def test_nested_enumerate(self, flags=force_pyobj_flags):
        self.run_nullary_func(nested_enumerate_usecase, flags)

    @tag('important')
    def test_nested_enumerate_npm(self):
        self.test_nested_enumerate(flags=no_pyobj_flags)

    def run_array_1d(self, item_type, arg, flags):
        # Iteration over a 1d numpy array
        pyfunc = scalar_iter_usecase
        cr = compile_isolated(pyfunc, (types.Array(item_type, 1, 'A'),),
                              item_type, flags=flags)
        cfunc = cr.entry_point
        self.assertPreciseEqual(cfunc(arg), pyfunc(arg))

    def test_array_1d_float(self, flags=force_pyobj_flags):
        self.run_array_1d(types.float64, np.arange(5.0), flags)

    def test_array_1d_float_npm(self):
        self.test_array_1d_float(no_pyobj_flags)

    def test_array_1d_complex(self, flags=force_pyobj_flags):
        self.run_array_1d(types.complex128, np.arange(5.0) * 1.0j, flags)

    @tag('important')
    def test_array_1d_complex_npm(self):
        self.test_array_1d_complex(no_pyobj_flags)

    def test_array_1d_record(self, flags=force_pyobj_flags):
        pyfunc = record_iter_usecase
        item_type = numpy_support.from_dtype(record_dtype)
        cr = compile_isolated(pyfunc, (types.Array(item_type, 1, 'A'),),
                              flags=flags)
        cfunc = cr.entry_point
        arr = np.recarray(3, dtype=record_dtype)
        for i in range(3):
            arr[i].a = float(i * 2)
            arr[i].b = i + 2
        got = pyfunc(arr)
        self.assertPreciseEqual(cfunc(arr), got)

    def test_array_1d_record_npm(self):
        self.test_array_1d_record(no_pyobj_flags)

    def test_array_1d_record_mutate_npm(self, flags=no_pyobj_flags):
        pyfunc = record_iter_mutate_usecase
        item_type = numpy_support.from_dtype(record_dtype)
        cr = compile_isolated(pyfunc, (types.Array(item_type, 1, 'A'),),
                              flags=flags)
        cfunc = cr.entry_point
        arr = np.recarray(3, dtype=record_dtype)
        for i in range(3):
            arr[i].a = float(i * 2)
            arr[i].b = i + 2
        expected = arr.copy()
        pyfunc(expected)
        got = arr.copy()
        cfunc(got)
        self.assertPreciseEqual(expected, got)

    def test_array_1d_record_mutate(self):
        self.test_array_1d_record_mutate_npm(flags=force_pyobj_flags)

    def test_tuple_iter_issue1504(self):
        # The issue is due to `row` being typed as heterogeneous tuple.
        def bar(x, y):
            total = 0
            for row in zip(x, y):
                total += row[0] + row[1]

            return total

        x = y = np.arange(3, dtype=np.int32)
        aryty = types.Array(types.int32, 1, 'C')
        cres = compile_isolated(bar, (aryty, aryty))

        expect = bar(x, y)
        got = cres.entry_point(x, y)
        self.assertEqual(expect, got)


if __name__ == '__main__':
    unittest.main()
