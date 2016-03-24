from __future__ import print_function

import itertools
import functools
import sys

import numpy as np

import numba.unittest_support as unittest
from numba.compiler import compile_isolated, Flags
from numba import jit, typeof, errors, types, utils
from .support import TestCase, tag


enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

forceobj_flags = Flags()
forceobj_flags.set("force_pyobject")

no_pyobj_flags = Flags()


def abs_usecase(x):
    return abs(x)

def all_usecase(x, y):
    if x == None and y == None:
        return all([])
    elif x == None:
        return all([y])
    elif y == None:
        return all([x])
    else:
        return all([x, y])

def any_usecase(x, y):
    if x == None and y == None:
        return any([])
    elif x == None:
        return any([y])
    elif y == None:
        return any([x])
    else:
        return any([x, y])

def bool_usecase(x):
    return bool(x)

def chr_usecase(x):
    return chr(x)

def cmp_usecase(x, y):
    return cmp(x, y)

def complex_usecase(x, y):
    return complex(x, y)

def enumerate_usecase():
    result = 0
    for i, j in enumerate((1., 2.5, 3.)):
        result += i * j
    return result

def enumerate_start_usecase():
    result = 0
    for i, j in enumerate((1., 2.5, 3.), 42):
        result += i * j
    return result

def filter_usecase(x, filter_func):
    return filter(filter_func, x)

def float_usecase(x):
    return float(x)

def format_usecase(x, y):
    return x.format(y)

def globals_usecase():
    return globals()

# NOTE: hash() is tested in test_hashing

def hex_usecase(x):
    return hex(x)

def int_usecase(x, base):
    return int(x, base=base)

def iter_next_usecase(x):
    it = iter(x)
    return next(it), next(it)

def locals_usecase(x):
    y = 5
    return locals()['y']

def long_usecase(x, base):
    return long(x, base=base)

def map_usecase(x, map_func):
    return map(map_func, x)

def max_usecase1(x, y):
    return max(x, y)

def max_usecase2(x, y):
    return max([x, y])

def max_usecase3(x):
    return max(x)

def min_usecase1(x, y):
    return min(x, y)

def min_usecase2(x, y):
    return min([x, y])

def min_usecase3(x):
    return min(x)

def oct_usecase(x):
    return oct(x)

def ord_usecase(x):
    return ord(x)

def reduce_usecase(reduce_func, x):
    return functools.reduce(reduce_func, x)

def round_usecase1(x):
    return round(x)

def round_usecase2(x, n):
    return round(x, n)

def sum_usecase(x):
    return sum(x)

def type_unary_usecase(a, b):
    return type(a)(b)

def unichr_usecase(x):
    return unichr(x)

def zip_usecase():
    result = 0
    for i, j in zip((1, 2, 3), (4.5, 6.7)):
        result += i * j
    return result

def zip_0_usecase():
    result = 0
    for i in zip():
        result += 1
    return result

def zip_1_usecase():
    result = 0
    for i, in zip((1, 2)):
        result += i
    return result


def zip_3_usecase():
    result = 0
    for i, j, k in zip((1, 2), (3, 4, 5), (6.7, 8.9)):
        result += i * j * k
    return result


def pow_op_usecase(x, y):
    return x ** y


def pow_usecase(x, y):
    return pow(x, y)


class TestBuiltins(TestCase):

    def run_nullary_func(self, pyfunc, flags):
        cr = compile_isolated(pyfunc, (), flags=flags)
        cfunc = cr.entry_point
        expected = pyfunc()
        self.assertPreciseEqual(cfunc(), expected)

    def test_abs(self, flags=enable_pyobj_flags):
        pyfunc = abs_usecase

        cr = compile_isolated(pyfunc, (types.int32,), flags=flags)
        cfunc = cr.entry_point
        for x in [-1, 0, 1]:
            self.assertPreciseEqual(cfunc(x), pyfunc(x))

        cr = compile_isolated(pyfunc, (types.float32,), flags=flags)
        cfunc = cr.entry_point
        for x in [-1.1, 0.0, 1.1]:
            self.assertPreciseEqual(cfunc(x), pyfunc(x), prec='single')

        complex_values = [-1.1 + 0.5j, 0.0 + 0j, 1.1 + 3j,
                          float('inf') + 1j * float('nan'),
                          float('nan') - 1j * float('inf')]
        cr = compile_isolated(pyfunc, (types.complex64,), flags=flags)
        cfunc = cr.entry_point
        for x in complex_values:
            self.assertPreciseEqual(cfunc(x), pyfunc(x), prec='single')
        cr = compile_isolated(pyfunc, (types.complex128,), flags=flags)
        cfunc = cr.entry_point
        for x in complex_values:
            self.assertPreciseEqual(cfunc(x), pyfunc(x))

    @tag('important')
    def test_abs_npm(self):
        self.test_abs(flags=no_pyobj_flags)

    def test_all(self, flags=enable_pyobj_flags):
        pyfunc = all_usecase

        cr = compile_isolated(pyfunc, (types.int32,types.int32), flags=flags)
        cfunc = cr.entry_point
        x_operands = [-1, 0, 1, None]
        y_operands = [-1, 0, 1, None]
        for x, y in itertools.product(x_operands, y_operands):
            self.assertPreciseEqual(cfunc(x, y), pyfunc(x, y))

    def test_all_npm(self):
        with self.assertTypingError():
            self.test_all(flags=no_pyobj_flags)

    def test_any(self, flags=enable_pyobj_flags):
        pyfunc = any_usecase

        cr = compile_isolated(pyfunc, (types.int32,types.int32), flags=flags)
        cfunc = cr.entry_point
        x_operands = [-1, 0, 1, None]
        y_operands = [-1, 0, 1, None]
        for x, y in itertools.product(x_operands, y_operands):
            self.assertPreciseEqual(cfunc(x, y), pyfunc(x, y))

    def test_any_npm(self):
        with self.assertTypingError():
            self.test_any(flags=no_pyobj_flags)

    def test_bool(self, flags=enable_pyobj_flags):
        pyfunc = bool_usecase

        cr = compile_isolated(pyfunc, (types.int32,), flags=flags)
        cfunc = cr.entry_point
        for x in [-1, 0, 1]:
            self.assertPreciseEqual(cfunc(x), pyfunc(x))
        cr = compile_isolated(pyfunc, (types.float64,), flags=flags)
        cfunc = cr.entry_point
        for x in [0.0, -0.0, 1.5, float('inf'), float('nan')]:
            self.assertPreciseEqual(cfunc(x), pyfunc(x))
        cr = compile_isolated(pyfunc, (types.complex128,), flags=flags)
        cfunc = cr.entry_point
        for x in [complex(0, float('inf')), complex(0, float('nan'))]:
            self.assertPreciseEqual(cfunc(x), pyfunc(x))

    def test_bool_npm(self):
        self.test_bool(flags=no_pyobj_flags)

    def test_bool_nonnumber(self, flags=enable_pyobj_flags):
        pyfunc = bool_usecase

        cr = compile_isolated(pyfunc, (types.string,), flags=flags)
        cfunc = cr.entry_point
        for x in ['x', '']:
            self.assertPreciseEqual(cfunc(x), pyfunc(x))

        cr = compile_isolated(pyfunc, (types.Dummy('list'),), flags=flags)
        cfunc = cr.entry_point
        for x in [[1], []]:
            self.assertPreciseEqual(cfunc(x), pyfunc(x))

    def test_bool_nonnumber_npm(self):
        with self.assertTypingError():
            self.test_bool_nonnumber(flags=no_pyobj_flags)

    def test_chr(self, flags=enable_pyobj_flags):
        pyfunc = chr_usecase

        cr = compile_isolated(pyfunc, (types.int32,), flags=flags)
        cfunc = cr.entry_point
        for x in range(256):
            self.assertPreciseEqual(cfunc(x), pyfunc(x))

    def test_chr_npm(self):
        with self.assertTypingError():
            self.test_chr(flags=no_pyobj_flags)

    @unittest.skipIf(utils.IS_PY3, "cmp not available as global is Py3")
    def test_cmp(self, flags=enable_pyobj_flags):
        pyfunc = cmp_usecase

        cr = compile_isolated(pyfunc, (types.int32, types.int32), flags=flags)
        cfunc = cr.entry_point

        x_operands = [-1, 0, 1]
        y_operands = [-1, 0, 1]
        for x, y in itertools.product(x_operands, y_operands):
            self.assertPreciseEqual(cfunc(x, y), pyfunc(x, y))

    @unittest.skipIf(utils.IS_PY3, "cmp not available as global is Py3")
    def test_cmp_npm(self):
        with self.assertTypingError():
            self.test_cmp(flags=no_pyobj_flags)

    def test_complex(self, flags=enable_pyobj_flags):
        pyfunc = complex_usecase

        cr = compile_isolated(pyfunc, (types.int32, types.int32), flags=flags)
        cfunc = cr.entry_point

        x_operands = [-1, 0, 1]
        y_operands = [-1, 0, 1]
        for x, y in itertools.product(x_operands, y_operands):
            self.assertPreciseEqual(cfunc(x, y), pyfunc(x, y))

    @tag('important')
    def test_complex_npm(self):
        self.test_complex(flags=no_pyobj_flags)

    def test_enumerate(self, flags=enable_pyobj_flags):
        self.run_nullary_func(enumerate_usecase, flags)

    def test_enumerate_npm(self):
        self.test_enumerate(flags=no_pyobj_flags)

    def test_enumerate_start(self, flags=enable_pyobj_flags):
        self.run_nullary_func(enumerate_start_usecase, flags)

    def test_enumerate_start_npm(self):
        self.test_enumerate_start(flags=no_pyobj_flags)

    def test_filter(self, flags=enable_pyobj_flags):
        pyfunc = filter_usecase
        cr = compile_isolated(pyfunc, (types.Dummy('list'),
                                       types.Dummy('function_ptr')),
                                       flags=flags)
        cfunc = cr.entry_point

        filter_func = lambda x: x % 2
        x = [0, 1, 2, 3, 4]
        self.assertSequenceEqual(list(cfunc(x, filter_func)),
                                 list(pyfunc(x, filter_func)))

    def test_filter_npm(self):
        with self.assertTypingError():
            self.test_filter(flags=no_pyobj_flags)

    def test_float(self, flags=enable_pyobj_flags):
        pyfunc = float_usecase

        cr = compile_isolated(pyfunc, (types.int32,), flags=flags)
        cfunc = cr.entry_point
        for x in [-1, 0, 1]:
            self.assertPreciseEqual(cfunc(x), pyfunc(x))

        cr = compile_isolated(pyfunc, (types.float32,), flags=flags)
        cfunc = cr.entry_point
        for x in [-1.1, 0.0, 1.1]:
            self.assertPreciseEqual(cfunc(x), pyfunc(x), prec='single')

        cr = compile_isolated(pyfunc, (types.string,), flags=flags)
        cfunc = cr.entry_point
        for x in ['-1.1', '0.0', '1.1']:
            self.assertPreciseEqual(cfunc(x), pyfunc(x))

    @tag('important')
    def test_float_npm(self):
        with self.assertTypingError():
            self.test_float(flags=no_pyobj_flags)

    def test_format(self, flags=enable_pyobj_flags):
        pyfunc = format_usecase

        cr = compile_isolated(pyfunc, (types.string, types.int32,), flags=flags)
        cfunc = cr.entry_point
        x = '{0}'
        for y in [-1, 0, 1]:
            self.assertPreciseEqual(cfunc(x, y), pyfunc(x, y))

        cr = compile_isolated(pyfunc, (types.string,
                                       types.float32,), flags=flags)
        cfunc = cr.entry_point
        x = '{0}'
        for y in [-1.1, 0.0, 1.1]:
            self.assertPreciseEqual(cfunc(x, y), pyfunc(x, y))

        cr = compile_isolated(pyfunc, (types.string,
                                       types.string,), flags=flags)
        cfunc = cr.entry_point
        x = '{0}'
        for y in ['a', 'b', 'c']:
            self.assertPreciseEqual(cfunc(x, y), pyfunc(x, y))

    def test_format_npm(self):
        with self.assertTypingError():
            self.test_format(flags=no_pyobj_flags)

    def test_globals(self, flags=enable_pyobj_flags):
        pyfunc = globals_usecase
        cr = compile_isolated(pyfunc, (), flags=flags)
        cfunc = cr.entry_point
        g = cfunc()
        self.assertIs(g, globals())

    def test_globals_npm(self):
        with self.assertTypingError():
            self.test_globals(flags=no_pyobj_flags)

    def test_globals_jit(self, **jit_flags):
        # Issue #416: weird behaviour of globals() in combination with
        # the @jit decorator.
        pyfunc = globals_usecase
        jitted = jit(**jit_flags)(pyfunc)
        self.assertIs(jitted(), globals())
        self.assertIs(jitted(), globals())

    def test_globals_jit_npm(self):
        with self.assertTypingError():
            self.test_globals_jit(nopython=True)

    def test_hex(self, flags=enable_pyobj_flags):
        pyfunc = hex_usecase

        cr = compile_isolated(pyfunc, (types.int32,), flags=flags)
        cfunc = cr.entry_point
        for x in [-1, 0, 1]:
            self.assertPreciseEqual(cfunc(x), pyfunc(x))

    def test_hex_npm(self):
        with self.assertTypingError():
            self.test_hex(flags=no_pyobj_flags)

    def test_int(self, flags=enable_pyobj_flags):
        pyfunc = int_usecase

        cr = compile_isolated(pyfunc, (types.string, types.int32), flags=flags)
        cfunc = cr.entry_point

        x_operands = ['-1', '0', '1', '10']
        y_operands = [2, 8, 10, 16]
        for x, y in itertools.product(x_operands, y_operands):
            self.assertPreciseEqual(cfunc(x, y), pyfunc(x, y))

    @tag('important')
    def test_int_npm(self):
        with self.assertTypingError():
            self.test_int(flags=no_pyobj_flags)

    def test_iter_next(self, flags=enable_pyobj_flags):
        pyfunc = iter_next_usecase
        cr = compile_isolated(pyfunc, (types.UniTuple(types.int32, 3),),
                              flags=flags)
        cfunc = cr.entry_point
        self.assertPreciseEqual(cfunc((1, 42, 5)), (1, 42))

        cr = compile_isolated(pyfunc, (types.UniTuple(types.int32, 1),),
                              flags=flags)
        cfunc = cr.entry_point
        with self.assertRaises(StopIteration):
            cfunc((1,))

    @tag('important')
    def test_iter_next_npm(self):
        self.test_iter_next(flags=no_pyobj_flags)

    def test_locals(self, flags=enable_pyobj_flags):
        pyfunc = locals_usecase
        with self.assertRaises(errors.ForbiddenConstruct):
            cr = compile_isolated(pyfunc, (types.int64,), flags=flags)

    def test_locals_forceobj(self):
        self.test_locals(flags=forceobj_flags)

    def test_locals_npm(self):
        with self.assertTypingError():
            self.test_locals(flags=no_pyobj_flags)

    @unittest.skipIf(utils.IS_PY3, "long is not available as global is Py3")
    def test_long(self, flags=enable_pyobj_flags):
        pyfunc = long_usecase

        cr = compile_isolated(pyfunc, (types.string, types.int64), flags=flags)
        cfunc = cr.entry_point

        x_operands = ['-1', '0', '1', '10']
        y_operands = [2, 8, 10, 16]
        for x, y in itertools.product(x_operands, y_operands):
            self.assertPreciseEqual(cfunc(x, y), pyfunc(x, y))

    @unittest.skipIf(utils.IS_PY3, "cmp not available as global is Py3")
    def test_long_npm(self):
        with self.assertTypingError():
            self.test_long(flags=no_pyobj_flags)

    def test_map(self, flags=enable_pyobj_flags):
        pyfunc = map_usecase
        cr = compile_isolated(pyfunc, (types.Dummy('list'),
                                       types.Dummy('function_ptr')),
                                       flags=flags)
        cfunc = cr.entry_point

        map_func = lambda x: x * 2
        x = [0, 1, 2, 3, 4]
        self.assertSequenceEqual(list(cfunc(x, map_func)),
                                 list(pyfunc(x, map_func)))

    def test_map_npm(self):
        with self.assertTypingError():
            self.test_map(flags=no_pyobj_flags)

    def test_max_1(self, flags=enable_pyobj_flags):
        pyfunc = max_usecase1
        cr = compile_isolated(pyfunc, (types.int32, types.int32), flags=flags)
        cfunc = cr.entry_point

        x_operands = [-1, 0, 1]
        y_operands = [-1, 0, 1]
        for x, y in itertools.product(x_operands, y_operands):
            self.assertPreciseEqual(cfunc(x, y), pyfunc(x, y))

    def test_max_2(self, flags=enable_pyobj_flags):
        pyfunc = max_usecase2
        cr = compile_isolated(pyfunc, (types.int32, types.int32), flags=flags)
        cfunc = cr.entry_point

        x_operands = [-1, 0, 1]
        y_operands = [-1, 0, 1]
        for x, y in itertools.product(x_operands, y_operands):
            self.assertPreciseEqual(cfunc(x, y), pyfunc(x, y))

    @tag('important')
    def test_max_npm_1(self):
        self.test_max_1(flags=no_pyobj_flags)

    def test_max_npm_2(self):
        with self.assertTypingError():
            self.test_max_2(flags=no_pyobj_flags)

    def test_min_1(self, flags=enable_pyobj_flags):
        pyfunc = min_usecase1
        cr = compile_isolated(pyfunc, (types.int32, types.int32), flags=flags)
        cfunc = cr.entry_point

        x_operands = [-1, 0, 1]
        y_operands = [-1, 0, 1]
        for x, y in itertools.product(x_operands, y_operands):
            self.assertPreciseEqual(cfunc(x, y), pyfunc(x, y))

    def test_min_2(self, flags=enable_pyobj_flags):
        pyfunc = min_usecase2
        cr = compile_isolated(pyfunc, (types.int32, types.int32), flags=flags)
        cfunc = cr.entry_point

        x_operands = [-1, 0, 1]
        y_operands = [-1, 0, 1]
        for x, y in itertools.product(x_operands, y_operands):
            self.assertPreciseEqual(cfunc(x, y), pyfunc(x, y))

    @tag('important')
    def test_min_npm_1(self):
        self.test_min_1(flags=no_pyobj_flags)

    def test_min_npm_2(self):
        with self.assertTypingError():
            self.test_min_2(flags=no_pyobj_flags)

    def check_min_max_invalid_types(self, pyfunc, flags=enable_pyobj_flags):
        cr = compile_isolated(pyfunc, (types.int32, types.Dummy('list')),
                              flags=flags)
        cfunc = cr.entry_point
        cfunc(1, [1])

    def test_max_1_invalid_types(self):
        # Heterogenous ordering is valid in Python 2
        if utils.IS_PY3:
            with self.assertRaises(TypeError):
                self.check_min_max_invalid_types(max_usecase1)
        else:
            self.check_min_max_invalid_types(max_usecase1)

    def test_max_1_invalid_types_npm(self):
        with self.assertTypingError():
            self.check_min_max_invalid_types(max_usecase1, flags=no_pyobj_flags)

    def test_min_1_invalid_types(self):
        # Heterogenous ordering is valid in Python 2
        if utils.IS_PY3:
            with self.assertRaises(TypeError):
                self.check_min_max_invalid_types(min_usecase1)
        else:
            self.check_min_max_invalid_types(min_usecase1)

    def test_min_1_invalid_types_npm(self):
        with self.assertTypingError():
            self.check_min_max_invalid_types(min_usecase1, flags=no_pyobj_flags)

    # Test that max(1) and min(1) fail

    def check_min_max_unary_non_iterable(self, pyfunc, flags=enable_pyobj_flags):
        cr = compile_isolated(pyfunc, (types.int32,), flags=flags)
        cfunc = cr.entry_point
        cfunc(1)

    def test_max_unary_non_iterable(self):
        with self.assertRaises(TypeError):
            self.check_min_max_unary_non_iterable(max_usecase3)

    def test_max_unary_non_iterable_npm(self):
        with self.assertTypingError():
            self.check_min_max_unary_non_iterable(max_usecase3)

    def test_min_unary_non_iterable(self):
        with self.assertRaises(TypeError):
            self.check_min_max_unary_non_iterable(min_usecase3)

    def test_min_unary_non_iterable_npm(self):
        with self.assertTypingError():
            self.check_min_max_unary_non_iterable(min_usecase3)

    def test_oct(self, flags=enable_pyobj_flags):
        pyfunc = oct_usecase

        cr = compile_isolated(pyfunc, (types.int32,), flags=flags)
        cfunc = cr.entry_point
        for x in [-8, -1, 0, 1, 8]:
            self.assertPreciseEqual(cfunc(x), pyfunc(x))

    def test_oct_npm(self):
        with self.assertTypingError():
            self.test_oct(flags=no_pyobj_flags)

    def test_ord(self, flags=enable_pyobj_flags):
        pyfunc = ord_usecase

        cr = compile_isolated(pyfunc, (types.string,), flags=flags)
        cfunc = cr.entry_point
        for x in ['a', u'\u2020']:
            self.assertPreciseEqual(cfunc(x), pyfunc(x))

    def test_ord_npm(self):
        with self.assertTypingError():
            self.test_ord(flags=no_pyobj_flags)

    def test_reduce(self, flags=enable_pyobj_flags):
        pyfunc = reduce_usecase
        cr = compile_isolated(pyfunc, (types.Dummy('function_ptr'),
                                       types.Dummy('list')),
                                       flags=flags)
        cfunc = cr.entry_point

        reduce_func = lambda x, y: x + y

        x = range(10)
        self.assertPreciseEqual(cfunc(reduce_func, x), pyfunc(reduce_func, x))

        x = [x + x/10.0 for x in range(10)]
        self.assertPreciseEqual(cfunc(reduce_func, x), pyfunc(reduce_func, x))

        x = [complex(x, x) for x in range(10)]
        self.assertPreciseEqual(cfunc(reduce_func, x), pyfunc(reduce_func, x))

    def test_reduce_npm(self):
        with self.assertTypingError():
            self.test_reduce(flags=no_pyobj_flags)

    # Under Windows, the LLVM "round" intrinsic (used for Python 2)
    # mistreats signed zeros.
    _relax_round = sys.platform == 'win32' and sys.version_info < (3,)

    def test_round1(self, flags=enable_pyobj_flags):
        pyfunc = round_usecase1

        for tp in (types.float64, types.float32):
            cr = compile_isolated(pyfunc, (tp,), flags=flags)
            cfunc = cr.entry_point
            values = [-1.6, -1.5, -1.4, -0.5, 0.0, 0.1, 0.5, 0.6, 1.4, 1.5, 5.0]
            if not self._relax_round:
                values += [-0.1, -0.0]
            for x in values:
                self.assertPreciseEqual(cfunc(x), pyfunc(x))

    def test_round1_npm(self):
        self.test_round1(flags=no_pyobj_flags)

    def test_round2(self, flags=enable_pyobj_flags):
        pyfunc = round_usecase2

        for tp in (types.float64, types.float32):
            prec = 'single' if tp is types.float32 else 'exact'
            cr = compile_isolated(pyfunc, (tp, types.int32), flags=flags)
            cfunc = cr.entry_point
            for x in [0.0, 0.1, 0.125, 0.25, 0.5, 0.75, 1.25,
                      1.5, 1.75, 2.25, 2.5, 2.75, 12.5, 15.0, 22.5]:
                for n in (-1, 0, 1, 2):
                    self.assertPreciseEqual(cfunc(x, n), pyfunc(x, n),
                                            prec=prec)
                    expected = pyfunc(-x, n)
                    if not (expected == 0.0 and self._relax_round):
                        self.assertPreciseEqual(cfunc(-x, n), pyfunc(-x, n),
                                                prec=prec)

    @tag('important')
    def test_round2_npm(self):
        self.test_round2(flags=no_pyobj_flags)

    def test_sum(self, flags=enable_pyobj_flags):
        pyfunc = sum_usecase

        cr = compile_isolated(pyfunc, (types.Dummy('list'),), flags=flags)
        cfunc = cr.entry_point

        x = range(10)
        self.assertPreciseEqual(cfunc(x), pyfunc(x))

        x = [x + x/10.0 for x in range(10)]
        self.assertPreciseEqual(cfunc(x), pyfunc(x))

        x = [complex(x, x) for x in range(10)]
        self.assertPreciseEqual(cfunc(x), pyfunc(x))

    def test_sum_npm(self):
        with self.assertTypingError():
            self.test_sum(flags=no_pyobj_flags)

    def test_type_unary(self):
        # Test type(val) and type(val)(other_val)
        pyfunc = type_unary_usecase
        cfunc = jit(nopython=True)(pyfunc)

        def check(*args):
            expected = pyfunc(*args)
            self.assertPreciseEqual(cfunc(*args), expected)

        check(1.5, 2)
        check(1, 2.5)
        check(1.5j, 2)
        check(True, 2)
        check(2.5j, False)

    @unittest.skipIf(utils.IS_PY3, "unichr not available as global is Py3")
    def test_unichr(self, flags=enable_pyobj_flags):
        pyfunc = unichr_usecase

        cr = compile_isolated(pyfunc, (types.int32,), flags=flags)
        cfunc = cr.entry_point
        for x in range(0, 1000, 10):
            self.assertPreciseEqual(cfunc(x), pyfunc(x))

    @unittest.skipIf(utils.IS_PY3, "unichr not available as global is Py3")
    def test_unichr_npm(self):
        with self.assertTypingError():
            self.test_unichr(flags=no_pyobj_flags)

    def test_zip(self, flags=forceobj_flags):
        self.run_nullary_func(zip_usecase, flags)

    @tag('important')
    def test_zip_npm(self):
        self.test_zip(flags=no_pyobj_flags)

    def test_zip_1(self, flags=forceobj_flags):
        self.run_nullary_func(zip_1_usecase, flags)

    @tag('important')
    def test_zip_1_npm(self):
        self.test_zip_1(flags=no_pyobj_flags)

    def test_zip_3(self, flags=forceobj_flags):
        self.run_nullary_func(zip_3_usecase, flags)

    @tag('important')
    def test_zip_3_npm(self):
        self.test_zip_3(flags=no_pyobj_flags)

    def test_zip_0(self, flags=forceobj_flags):
        self.run_nullary_func(zip_0_usecase, flags)

    def test_zip_0_npm(self):
        self.test_zip_0(flags=no_pyobj_flags)

    def test_pow_op_usecase(self):
        args = [
            (2, 3),
            (2.0, 3),
            (2, 3.0),
            (2j, 3.0j),
        ]

        for x, y in args:
            cres = compile_isolated(pow_op_usecase, (typeof(x), typeof(y)),
                                    flags=no_pyobj_flags)
            r = cres.entry_point(x, y)
            self.assertPreciseEqual(r, pow_op_usecase(x, y))

    @tag('important')
    def test_pow_usecase(self):
        args = [
            (2, 3),
            (2.0, 3),
            (2, 3.0),
            (2j, 3.0j),
        ]

        for x, y in args:
            cres = compile_isolated(pow_usecase, (typeof(x), typeof(y)),
                                    flags=no_pyobj_flags)
            r = cres.entry_point(x, y)
            self.assertPreciseEqual(r, pow_usecase(x, y))


if __name__ == '__main__':
    unittest.main()
