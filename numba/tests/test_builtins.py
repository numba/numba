import itertools
import functools
import sys
import operator

import numpy as np

import unittest
from numba.core.compiler import compile_isolated, Flags
from numba import jit, typeof, njit
from numba.core import errors, types, utils, config
from numba.tests.support import TestCase, tag


enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

forceobj_flags = Flags()
forceobj_flags.set("force_pyobject")

no_pyobj_flags = Flags()

nrt_no_pyobj_flags = Flags()
nrt_no_pyobj_flags.set("nrt")


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

def complex_usecase(x, y):
    return complex(x, y)

def divmod_usecase(x, y):
    return divmod(x, y)

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

def enumerate_invalid_start_usecase():
    result = 0
    for i, j in enumerate((1., 2.5, 3.), 3.14159):
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

def max_usecase4():
    return max(())


def min_usecase1(x, y):
    return min(x, y)

def min_usecase2(x, y):
    return min([x, y])

def min_usecase3(x):
    return min(x)

def min_usecase4():
    return min(())


def oct_usecase(x):
    return oct(x)

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

def truth_usecase(p):
    return operator.truth(p)

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


def zip_first_exhausted():
    iterable = range(7)
    n = 3
    it = iter(iterable)
    # 1st iterator is shorter
    front = list(zip(range(n), it))
    # Make sure that we didn't skip one in `it`
    back = list(it)
    return front, back


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

        for unsigned_type in types.unsigned_domain:
            unsigned_values = [0, 10, 2, 2 ** unsigned_type.bitwidth - 1]
            cr = compile_isolated(pyfunc, (unsigned_type,), flags=flags)
            cfunc = cr.entry_point
            for x in unsigned_values:
                self.assertPreciseEqual(cfunc(x), pyfunc(x))

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

    def test_complex(self, flags=enable_pyobj_flags):
        pyfunc = complex_usecase

        cr = compile_isolated(pyfunc, (types.int32, types.int32), flags=flags)
        cfunc = cr.entry_point

        x_operands = [-1, 0, 1]
        y_operands = [-1, 0, 1]
        for x, y in itertools.product(x_operands, y_operands):
            self.assertPreciseEqual(cfunc(x, y), pyfunc(x, y))

    def test_complex_npm(self):
        self.test_complex(flags=no_pyobj_flags)

    def test_divmod_ints(self, flags=enable_pyobj_flags):
        pyfunc = divmod_usecase

        cr = compile_isolated(pyfunc, (types.int64, types.int64),
                              flags=flags)
        cfunc = cr.entry_point

        def truncate_result(x, bits=64):
            # Remove any extraneous bits (since Numba will return
            # a 64-bit result by definition)
            if x >= 0:
                x &= (1 << (bits - 1)) - 1
            return x

        denominators = [1, 3, 7, 15, -1, -3, -7, -15, 2**63 - 1, -2**63]
        numerators = denominators + [0]
        for x, y, in itertools.product(numerators, denominators):
            expected_quot, expected_rem = pyfunc(x, y)
            quot, rem = cfunc(x, y)
            f = truncate_result
            self.assertPreciseEqual((f(quot), f(rem)),
                                    (f(expected_quot), f(expected_rem)))

        for x in numerators:
            with self.assertRaises(ZeroDivisionError):
                cfunc(x, 0)

    def test_divmod_ints_npm(self):
        self.test_divmod_ints(flags=no_pyobj_flags)

    def test_divmod_floats(self, flags=enable_pyobj_flags):
        pyfunc = divmod_usecase

        cr = compile_isolated(pyfunc, (types.float64, types.float64),
                              flags=flags)
        cfunc = cr.entry_point

        denominators = [1., 3.5, 1e100, -2., -7.5, -1e101,
                        np.inf, -np.inf, np.nan]
        numerators = denominators + [-0.0, 0.0]
        for x, y, in itertools.product(numerators, denominators):
            expected_quot, expected_rem = pyfunc(x, y)
            quot, rem = cfunc(x, y)
            self.assertPreciseEqual((quot, rem), (expected_quot, expected_rem))

        for x in numerators:
            with self.assertRaises(ZeroDivisionError):
                cfunc(x, 0.0)

    def test_divmod_floats_npm(self):
        self.test_divmod_floats(flags=no_pyobj_flags)

    def test_enumerate(self, flags=enable_pyobj_flags):
        self.run_nullary_func(enumerate_usecase, flags)

    def test_enumerate_npm(self):
        self.test_enumerate(flags=no_pyobj_flags)

    def test_enumerate_start(self, flags=enable_pyobj_flags):
        self.run_nullary_func(enumerate_start_usecase, flags)

    def test_enumerate_start_npm(self):
        self.test_enumerate_start(flags=no_pyobj_flags)

    def test_enumerate_start_invalid_start_type(self):
        pyfunc = enumerate_invalid_start_usecase
        cr = compile_isolated(pyfunc, (), flags=enable_pyobj_flags)
        with self.assertRaises(TypeError) as raises:
            cr.entry_point()

        msg = "'float' object cannot be interpreted as an integer"
        self.assertIn(msg, str(raises.exception))

    def test_enumerate_start_invalid_start_type_npm(self):
        pyfunc = enumerate_invalid_start_usecase
        with self.assertRaises(errors.TypingError) as raises:
            cr = compile_isolated(pyfunc, (), flags=no_pyobj_flags)
        msg = "Only integers supported as start value in enumerate"
        self.assertIn(msg, str(raises.exception))

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

    #
    # min() and max()
    #

    def check_minmax_1(self, pyfunc, flags):
        cr = compile_isolated(pyfunc, (types.int32, types.int32), flags=flags)
        cfunc = cr.entry_point

        x_operands = [-1, 0, 1]
        y_operands = [-1, 0, 1]
        for x, y in itertools.product(x_operands, y_operands):
            self.assertPreciseEqual(cfunc(x, y), pyfunc(x, y))

    def test_max_1(self, flags=enable_pyobj_flags):
        """
        max(*args)
        """
        self.check_minmax_1(max_usecase1, flags)

    def test_min_1(self, flags=enable_pyobj_flags):
        """
        min(*args)
        """
        self.check_minmax_1(min_usecase1, flags)

    def test_max_npm_1(self):
        self.test_max_1(flags=no_pyobj_flags)

    def test_min_npm_1(self):
        self.test_min_1(flags=no_pyobj_flags)

    def check_minmax_2(self, pyfunc, flags):
        cr = compile_isolated(pyfunc, (types.int32, types.int32), flags=flags)
        cfunc = cr.entry_point

        x_operands = [-1, 0, 1]
        y_operands = [-1, 0, 1]
        for x, y in itertools.product(x_operands, y_operands):
            self.assertPreciseEqual(cfunc(x, y), pyfunc(x, y))

    def test_max_2(self, flags=enable_pyobj_flags):
        """
        max(list)
        """
        self.check_minmax_2(max_usecase2, flags)

    def test_min_2(self, flags=enable_pyobj_flags):
        """
        min(list)
        """
        self.check_minmax_2(min_usecase2, flags)

    def test_max_npm_2(self):
        with self.assertTypingError():
            self.test_max_2(flags=no_pyobj_flags)

    def test_min_npm_2(self):
        with self.assertTypingError():
            self.test_min_2(flags=no_pyobj_flags)

    def check_minmax_3(self, pyfunc, flags):
        def check(argty):
            cr = compile_isolated(pyfunc, (argty,), flags=flags)
            cfunc = cr.entry_point
            # Check that the algorithm matches Python's with a non-total order
            tup = (1.5, float('nan'), 2.5)
            for val in [tup, tup[::-1]]:
                self.assertPreciseEqual(cfunc(val), pyfunc(val))

        check(types.UniTuple(types.float64, 3))
        check(types.Tuple((types.float32, types.float64, types.float32)))

    def test_max_3(self, flags=enable_pyobj_flags):
        """
        max(tuple)
        """
        self.check_minmax_3(max_usecase3, flags)

    def test_min_3(self, flags=enable_pyobj_flags):
        """
        min(tuple)
        """
        self.check_minmax_3(min_usecase3, flags)

    def test_max_npm_3(self):
        self.test_max_3(flags=no_pyobj_flags)

    def test_min_npm_3(self):
        self.test_min_3(flags=no_pyobj_flags)

    def check_min_max_invalid_types(self, pyfunc, flags=enable_pyobj_flags):
        cr = compile_isolated(pyfunc, (types.int32, types.Dummy('list')),
                              flags=flags)
        cfunc = cr.entry_point
        cfunc(1, [1])

    def test_max_1_invalid_types(self):
        with self.assertRaises(TypeError):
            self.check_min_max_invalid_types(max_usecase1)

    def test_max_1_invalid_types_npm(self):
        with self.assertTypingError():
            self.check_min_max_invalid_types(max_usecase1, flags=no_pyobj_flags)

    def test_min_1_invalid_types(self):
        with self.assertRaises(TypeError):
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

    # Test that max(()) and min(()) fail

    def check_min_max_empty_tuple(self, pyfunc, func_name):
        with self.assertTypingError() as raises:
            compile_isolated(pyfunc, (), flags=no_pyobj_flags)
        self.assertIn("%s() argument is an empty tuple" % func_name,
                      str(raises.exception))

    def test_max_empty_tuple(self):
        self.check_min_max_empty_tuple(max_usecase4, "max")

    def test_min_empty_tuple(self):
        self.check_min_max_empty_tuple(min_usecase4, "min")


    def test_oct(self, flags=enable_pyobj_flags):
        pyfunc = oct_usecase

        cr = compile_isolated(pyfunc, (types.int32,), flags=flags)
        cfunc = cr.entry_point
        for x in [-8, -1, 0, 1, 8]:
            self.assertPreciseEqual(cfunc(x), pyfunc(x))

    def test_oct_npm(self):
        with self.assertTypingError():
            self.test_oct(flags=no_pyobj_flags)

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

    def test_round1(self, flags=enable_pyobj_flags):
        pyfunc = round_usecase1

        for tp in (types.float64, types.float32):
            cr = compile_isolated(pyfunc, (tp,), flags=flags)
            cfunc = cr.entry_point
            values = [-1.6, -1.5, -1.4, -0.5, 0.0, 0.1, 0.5, 0.6, 1.4, 1.5, 5.0]
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
                    self.assertPreciseEqual(cfunc(-x, n), pyfunc(-x, n),
                                            prec=prec)

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

    def test_truth(self):
        pyfunc = truth_usecase
        cfunc = jit(nopython=True)(pyfunc)

        self.assertEqual(pyfunc(True), cfunc(True))
        self.assertEqual(pyfunc(False), cfunc(False))

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

    def test_zip(self, flags=forceobj_flags):
        self.run_nullary_func(zip_usecase, flags)

    def test_zip_npm(self):
        self.test_zip(flags=no_pyobj_flags)

    def test_zip_1(self, flags=forceobj_flags):
        self.run_nullary_func(zip_1_usecase, flags)

    def test_zip_1_npm(self):
        self.test_zip_1(flags=no_pyobj_flags)

    def test_zip_3(self, flags=forceobj_flags):
        self.run_nullary_func(zip_3_usecase, flags)

    def test_zip_3_npm(self):
        self.test_zip_3(flags=no_pyobj_flags)

    def test_zip_0(self, flags=forceobj_flags):
        self.run_nullary_func(zip_0_usecase, flags)

    def test_zip_0_npm(self):
        self.test_zip_0(flags=no_pyobj_flags)

    def test_zip_first_exhausted(self, flags=forceobj_flags):
        """
        Test side effect to the input iterators when a left iterator has been
        exhausted before the ones on the right.
        """
        self.run_nullary_func(zip_first_exhausted, flags)

    def test_zip_first_exhausted_npm(self):
        self.test_zip_first_exhausted(flags=nrt_no_pyobj_flags)

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

    def _check_min_max(self, pyfunc):
        cfunc = njit()(pyfunc)
        expected = pyfunc()
        got = cfunc()
        self.assertPreciseEqual(expected, got)

    def test_min_max_iterable_input(self):

        @njit
        def frange(start, stop, step):
            i = start
            while i < stop:
                yield i
                i += step

        def sample_functions(op):
            yield lambda: op(range(10))
            yield lambda: op(range(4, 12))
            yield lambda: op(range(-4, -15, -1))
            yield lambda: op([6.6, 5.5, 7.7])
            yield lambda: op([(3, 4), (1, 2)])
            yield lambda: op(frange(1.1, 3.3, 0.1))
            yield lambda: op([np.nan, -np.inf, np.inf, np.nan])
            yield lambda: op([(3,), (1,), (2,)])

        for fn in sample_functions(op=min):
            self._check_min_max(fn)

        for fn in sample_functions(op=max):
            self._check_min_max(fn)


class TestOperatorMixedTypes(TestCase):

    def test_eq_ne(self):
        for opstr in ('eq', 'ne'):
            op = getattr(operator, opstr)

            @njit
            def func(a, b):
                return op(a, b)

            # all these things should evaluate to being equal or not, all should
            # survive typing.
            things = (1, 0, True, False, 1.0, 2.0, 1.1, 1j, None, "", "1")
            for x, y in itertools.product(things, things):
                self.assertPreciseEqual(func.py_func(x, y), func(x, y))

    def test_cmp(self):
        for opstr in ('gt', 'lt', 'ge', 'le', 'eq', 'ne'):
            op = getattr(operator, opstr)
            @njit
            def func(a, b):
                return op(a, b)

            # numerical things should all be comparable
            things = (1, 0, True, False, 1.0, 0.0, 1.1)
            for x, y in itertools.product(things, things):
                expected = func.py_func(x, y)
                got = func(x, y)
                self.assertEqual(expected, got)


if __name__ == '__main__':
    unittest.main()
