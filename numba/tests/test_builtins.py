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

def enumerate_usecase():
    result = 0
    for i, j in enumerate([1,2,3]):
        result += i * j
    return result


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

    def test_enumerate(self, flags=enable_pyobj_flags):
        pyfunc = enumerate_usecase
        cr = compile_isolated(pyfunc, (), flags=flags)
        cfunc = cr.entry_point
        self.assertEqual(cfunc(), pyfunc())

    @unittest.expectedFailure
    def test_enumerate_npm(self):
        self.test_enumerate(flags=no_pyobj_flags)


if __name__ == '__main__':
    unittest.main()

