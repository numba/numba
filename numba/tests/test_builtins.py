from __future__ import print_function
import numba.unittest_support as unittest
from numba.compiler import compile_isolated, Flags
from numba import types, utils
import itertools

enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

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
    for i, j in enumerate([1,2,3]):
        result += i * j
    return result

def filter_usecase(x, filter_func):
    return filter(filter_func, x)

def float_usecase(x):
    return float(x)

def format_usecase(x, y):
    return x.format(y)

def hex_usecase(x):
    return hex(x)

def int_usecase(x, base):
    return int(x, base=base)

def long_usecase(x, base):
    return long(x, base=base)

def map_usecase(x, map_func):
    return map(map_func, x)


class TestBuiltins(unittest.TestCase):

    def test_abs(self, flags=enable_pyobj_flags):
        pyfunc = abs_usecase

        cr = compile_isolated(pyfunc, (types.int32,), flags=flags)
        cfunc = cr.entry_point
        for x in [-1, 0, 1]:
            self.assertEqual(cfunc(x), pyfunc(x))
    
        cr = compile_isolated(pyfunc, (types.float32,), flags=flags)
        cfunc = cr.entry_point
        for x in [-1.1, 0.0, 1.1]:
            self.assertAlmostEqual(cfunc(x), pyfunc(x))

    def test_abs_npm(self):
        self.test_abs(flags=no_pyobj_flags)
    
    def test_all(self, flags=enable_pyobj_flags):
        pyfunc = all_usecase

        cr = compile_isolated(pyfunc, (types.int32,types.int32), flags=flags)
        cfunc = cr.entry_point
        x_operands = [-1, 0, 1, None]
        y_operands = [-1, 0, 1, None]
        for x, y in itertools.product(x_operands, y_operands):
            self.assertEqual(cfunc(x, y), pyfunc(x, y))
        
    @unittest.expectedFailure
    def test_all_npm(self):
        self.test_all(flags=no_pyobj_flags)
    
    def test_any(self, flags=enable_pyobj_flags):
        pyfunc = any_usecase

        cr = compile_isolated(pyfunc, (types.int32,types.int32), flags=flags)
        cfunc = cr.entry_point
        x_operands = [-1, 0, 1, None]
        y_operands = [-1, 0, 1, None]
        for x, y in itertools.product(x_operands, y_operands):
            self.assertEqual(cfunc(x, y), pyfunc(x, y))
        
    @unittest.expectedFailure
    def test_any_npm(self):
        self.test_any(flags=no_pyobj_flags)
    
    def test_bool(self, flags=enable_pyobj_flags):
        pyfunc = bool_usecase

        cr = compile_isolated(pyfunc, (types.int32,), flags=flags)
        cfunc = cr.entry_point
        for x in [-1, 0, 1]:
            self.assertEqual(cfunc(x), pyfunc(x))

    def test_bool_npm(self):
        self.test_bool(flags=no_pyobj_flags)

    def test_bool_nonnumber(self, flags=enable_pyobj_flags):
        pyfunc = bool_usecase

        cr = compile_isolated(pyfunc, (types.string,), flags=flags)
        cfunc = cr.entry_point
        for x in ['x', '']:
            self.assertEqual(cfunc(x), pyfunc(x))

        cr = compile_isolated(pyfunc, (types.Dummy('list'),), flags=flags)
        cfunc = cr.entry_point
        for x in [[1], []]:
            self.assertEqual(cfunc(x), pyfunc(x))

    @unittest.expectedFailure
    def test_bool_nonnumber_npm(self):
        self.test_bool_nonnumber(flags=no_pyobj_flags)

    def test_chr(self, flags=enable_pyobj_flags):
        pyfunc = chr_usecase

        cr = compile_isolated(pyfunc, (types.int32,), flags=flags)
        cfunc = cr.entry_point
        for x in range(256):
            self.assertEqual(cfunc(x), pyfunc(x))

    @unittest.expectedFailure
    def test_chr_npm(self):
        self.test_chr(flags=no_pyobj_flags)

    def test_cmp(self, flags=enable_pyobj_flags):
        pyfunc = cmp_usecase

        cr = compile_isolated(pyfunc, (types.int32, types.int32), flags=flags)
        cfunc = cr.entry_point
        
        x_operands = [-1, 0, 1]
        y_operands = [-1, 0, 1]
        for x, y in itertools.product(x_operands, y_operands):
            self.assertEqual(cfunc(x, y), pyfunc(x, y))

    @unittest.expectedFailure
    def test_cmp_npm(self):
        self.test_cmp(flags=no_pyobj_flags)

    def test_complex(self, flags=enable_pyobj_flags):
        pyfunc = complex_usecase

        cr = compile_isolated(pyfunc, (types.int32, types.int32), flags=flags)
        cfunc = cr.entry_point
        
        x_operands = [-1, 0, 1]
        y_operands = [-1, 0, 1]
        for x, y in itertools.product(x_operands, y_operands):
            self.assertEqual(cfunc(x, y), pyfunc(x, y))

    def test_complex_npm(self):
        self.test_complex(flags=no_pyobj_flags)

    def test_enumerate(self, flags=enable_pyobj_flags):
        pyfunc = enumerate_usecase
        cr = compile_isolated(pyfunc, (), flags=flags)
        cfunc = cr.entry_point
        self.assertEqual(cfunc(), pyfunc())

    @unittest.expectedFailure
    def test_enumerate_npm(self):
        self.test_enumerate(flags=no_pyobj_flags)

    def test_filter(self, flags=enable_pyobj_flags):
        pyfunc = filter_usecase
        cr = compile_isolated(pyfunc, (types.Dummy('list'),
                                       types.Dummy('function_ptr')),
                                       flags=flags)
        cfunc = cr.entry_point

        filter_func = lambda x: x % 2
        x = [0, 1, 2, 3, 4]
        self.assertEqual(cfunc(x, filter_func), pyfunc(x, filter_func))

    @unittest.expectedFailure
    def test_filter_npm(self):
        self.test_filter(flags=no_pyobj_flags)

    def test_float(self, flags=enable_pyobj_flags):
        pyfunc = float_usecase

        cr = compile_isolated(pyfunc, (types.int32,), flags=flags)
        cfunc = cr.entry_point
        for x in [-1, 0, 1]:
            self.assertAlmostEqual(cfunc(x), pyfunc(x))

        cr = compile_isolated(pyfunc, (types.float32,), flags=flags)
        cfunc = cr.entry_point
        for x in [-1.1, 0.0, 1.1]:
            self.assertAlmostEqual(cfunc(x), pyfunc(x))

        cr = compile_isolated(pyfunc, (types.string,), flags=flags)
        cfunc = cr.entry_point
        for x in ['-1.1', '0.0', '1.1']:
            self.assertAlmostEqual(cfunc(x), pyfunc(x))

    @unittest.expectedFailure
    def test_float_npm(self):
        self.test_float(flags=no_pyobj_flags)

    def test_format(self, flags=enable_pyobj_flags):
        pyfunc = format_usecase

        cr = compile_isolated(pyfunc, (types.string,types.int32,), flags=flags)
        cfunc = cr.entry_point
        x = '{0}'
        for y in [-1, 0, 1]:
            self.assertAlmostEqual(cfunc(x, y), pyfunc(x, y))

        cr = compile_isolated(pyfunc, (types.string,
                                       types.float32,), flags=flags)
        cfunc = cr.entry_point
        x = '{0}'
        for y in [-1.1, 0.0, 1.1]:
            self.assertAlmostEqual(cfunc(x, y), pyfunc(x, y))

        cr = compile_isolated(pyfunc, (types.string,
                                       types.string,), flags=flags)
        cfunc = cr.entry_point
        x = '{0}'
        for y in ['a', 'b', 'c']:
            self.assertAlmostEqual(cfunc(x, y), pyfunc(x, y))

    @unittest.expectedFailure
    def test_format_npm(self):
        self.test_format(flags=no_pyobj_flags)

    def test_hex(self, flags=enable_pyobj_flags):
        pyfunc = hex_usecase

        cr = compile_isolated(pyfunc, (types.int32,), flags=flags)
        cfunc = cr.entry_point
        for x in [-1, 0, 1]:
            self.assertEqual(cfunc(x), pyfunc(x))

    @unittest.expectedFailure
    def test_hex_npm(self):
        self.test_hex(flags=no_pyobj_flags)

    def test_int(self, flags=enable_pyobj_flags):
        pyfunc = int_usecase

        cr = compile_isolated(pyfunc, (types.string, types.int32), flags=flags)
        cfunc = cr.entry_point

        x_operands = ['-1', '0', '1', '10']
        y_operands = [2, 8, 10, 16]
        for x, y in itertools.product(x_operands, y_operands):
            self.assertEqual(cfunc(x, y), pyfunc(x, y))

    @unittest.expectedFailure
    def test_int_npm(self):
        self.test_int(flags=no_pyobj_flags)

    def test_long(self, flags=enable_pyobj_flags):
        pyfunc = long_usecase

        cr = compile_isolated(pyfunc, (types.string, types.int64), flags=flags)
        cfunc = cr.entry_point

        x_operands = ['-1', '0', '1', '10']
        y_operands = [2, 8, 10, 16]
        for x, y in itertools.product(x_operands, y_operands):
            self.assertEqual(cfunc(x, y), pyfunc(x, y))

    @unittest.expectedFailure
    def test_long_npm(self):
        self.test_long(flags=no_pyobj_flags)

    def test_map(self, flags=enable_pyobj_flags):
        pyfunc = map_usecase
        cr = compile_isolated(pyfunc, (types.Dummy('list'),
                                       types.Dummy('function_ptr')),
                                       flags=flags)
        cfunc = cr.entry_point

        map_func = lambda x: x * 2
        x = [0, 1, 2, 3, 4]
        self.assertEqual(cfunc(x, map_func), pyfunc(x, map_func))

    @unittest.expectedFailure
    def test_map_npm(self):
        self.test_map(flags=no_pyobj_flags)


if __name__ == '__main__':
    unittest.main()

