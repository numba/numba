from __future__ import print_function

import itertools
import warnings

import numpy as np

from numba.compiler import compile_isolated, Flags
from numba import types, typeinfer
from numba.config import PYVERSION
from numba.tests.true_div_usecase import truediv_usecase
import numba.unittest_support as unittest

Noflags = Flags()

enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")


def add_usecase(x, y):
    return x + y

def sub_usecase(x, y):
    return x - y

def mul_usecase(x, y):
    return x * y

def div_usecase(x, y):
    return x / y

def floordiv_usecase(x, y):
    return x / y

def mod_usecase(x, y):
    return x % y

def pow_usecase(x, y):
    return x ** y

def bitshift_left_usecase(x, y):
    return x << y

def bitshift_right_usecase(x, y):
    return x >> y

def bitwise_and_usecase(x, y):
    return x & y

def bitwise_or_usecase(x, y):
    return x | y

def bitwise_xor_usecase(x, y):
    return x ^ y

def bitwise_not_usecase(x, y):
    return ~x

def not_usecase(x):
    return not(x)

def negate_usecase(x):
    return -x

def unary_positive_usecase(x):
    return +x

def lt_usecase(x, y):
    return x < y

def le_usecase(x, y):
    return x <= y

def gt_usecase(x, y):
    return x > y

def ge_usecase(x, y):
    return x >= y

def eq_usecase(x, y):
    return x == y

def ne_usecase(x, y):
    return x != y


class TestOperators(unittest.TestCase):

    def run_test_ints(self, pyfunc, x_operands, y_operands, types_list,
                      flags=enable_pyobj_flags):
        for arg_types in types_list:
            cr = compile_isolated(pyfunc, arg_types, flags=flags)
            cfunc = cr.entry_point

            for x, y in itertools.product(x_operands, y_operands):
                self.assertTrue(np.all(pyfunc(x, y) == cfunc(x, y)))

    def run_test_floats(self, pyfunc, x_operands, y_operands, types_list,
                        flags=enable_pyobj_flags):
        for arg_types in types_list:
            cr = compile_isolated(pyfunc, arg_types, flags=flags)
            cfunc = cr.entry_point
            for x, y in itertools.product(x_operands, y_operands):
                self.assertTrue(np.allclose(pyfunc(x, y), cfunc(x, y)))

    def coerce_operand(self, op, numba_type):
        if hasattr(op, "dtype"):
            return numba_type.cast_python_value(op)
        elif numba_type in types.unsigned_domain:
            return abs(int(op.real))
        elif numba_type in types.integer_domain:
            return int(op.real)
        elif numba_type in types.real_domain:
            return float(op.real)
        else:
            return op

    def run_test_scalar_compare(self, pyfunc, flags=enable_pyobj_flags,
                                ordered=True):
        ops = self.compare_scalar_operands
        types_list = self.compare_types
        if not ordered:
            types_list = types_list + self.compare_unordered_types
        for typ in types_list:
            cr = compile_isolated(pyfunc, (typ, typ), flags=flags)
            cfunc = cr.entry_point
            for x, y in itertools.product(ops, ops):
                x = self.coerce_operand(x, typ)
                y = self.coerce_operand(y, typ)
                expected = pyfunc(x, y)
                got = cfunc(x, y)
                # Scalar ops => scalar result
                self.assertIs(type(got), type(expected))
                self.assertEqual(got, expected,
                                 "mismatch with %r (%r, %r)"
                                 % (typ, x, y))

    def run_test_array_compare(self, pyfunc, flags=enable_pyobj_flags,
                               ordered=True):
        ops = np.array(self.compare_scalar_operands)
        types_list = self.compare_types
        if not ordered:
            types_list = types_list + self.compare_unordered_types
        for typ in types_list:
            array_type = types.Array(typ, 1, 'C')
            cr = compile_isolated(pyfunc, (array_type, array_type),
                                  flags=flags)
            cfunc = cr.entry_point
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', np.ComplexWarning)
                arr = typ.cast_python_value(ops)
            for i in range(len(arr)):
                x = arr
                y = np.concatenate((arr[i:], arr[:i]))
                expected = pyfunc(x, y)
                got = cfunc(x, y)
                # Array ops => array result
                self.assertEqual(got.dtype, expected.dtype)
                self.assertTrue(np.all(got == expected),
                                "mismatch with %r (%r, %r): %r != %r"
                                % (typ, x, y, got, expected))

    compare_scalar_operands = [-0.5, -1.0 + 1j, -1.0 + 2j, -0.5 + 1j, 1.5]
    compare_types = [types.int32, types.int64,
                     types.uint32, types.uint64,
                     types.float32, types.float64]
    compare_unordered_types = [types.complex64, types.complex128]

    def test_lt_scalar(self, flags=enable_pyobj_flags):
        self.run_test_scalar_compare(lt_usecase, flags)

    def test_lt_scalar_npm(self):
        self.test_lt_scalar(flags=Noflags)

    def test_le_scalar(self, flags=enable_pyobj_flags):
        self.run_test_scalar_compare(le_usecase, flags)

    def test_le_scalar_npm(self):
        self.test_le_scalar(flags=Noflags)

    def test_gt_scalar(self, flags=enable_pyobj_flags):
        self.run_test_scalar_compare(gt_usecase, flags)

    def test_gt_scalar_npm(self):
        self.test_gt_scalar(flags=Noflags)

    def test_ge_scalar(self, flags=enable_pyobj_flags):
        self.run_test_scalar_compare(ge_usecase, flags)

    def test_ge_scalar_npm(self):
        self.test_ge_scalar(flags=Noflags)

    def test_eq_scalar(self, flags=enable_pyobj_flags):
        self.run_test_scalar_compare(eq_usecase, flags, ordered=False)

    def test_eq_scalar_npm(self):
        self.test_eq_scalar(flags=Noflags)

    def test_ne_scalar(self, flags=enable_pyobj_flags):
        self.run_test_scalar_compare(ne_usecase, flags, ordered=False)

    def test_ne_scalar_npm(self):
        self.test_ne_scalar(flags=Noflags)

    def test_eq_array(self, flags=enable_pyobj_flags):
        self.run_test_array_compare(eq_usecase, flags, ordered=False)

    @unittest.expectedFailure
    def test_eq_array_npm(self):
        self.test_eq_array(flags=Noflags)

    def test_ne_array(self, flags=enable_pyobj_flags):
        self.run_test_array_compare(ne_usecase, flags, ordered=False)

    @unittest.expectedFailure
    def test_ne_array_npm(self):
        self.test_ne_array(flags=Noflags)

    def test_lt_array(self, flags=enable_pyobj_flags):
        self.run_test_array_compare(lt_usecase, flags)

    @unittest.expectedFailure
    def test_lt_array_npm(self):
        self.test_lt_array(flags=Noflags)

    def test_le_array(self, flags=enable_pyobj_flags):
        self.run_test_array_compare(le_usecase, flags)

    @unittest.expectedFailure
    def test_le_array_npm(self):
        self.test_le_array(flags=Noflags)

    def test_gt_array(self, flags=enable_pyobj_flags):
        self.run_test_array_compare(gt_usecase, flags)

    @unittest.expectedFailure
    def test_gt_array_npm(self):
        self.test_gt_array(flags=Noflags)

    def test_ge_array(self, flags=enable_pyobj_flags):
        self.run_test_array_compare(ge_usecase, flags)

    @unittest.expectedFailure
    def test_ge_array_npm(self):
        self.test_ge_array(flags=Noflags)

    def test_add_ints(self, flags=enable_pyobj_flags):

        pyfunc = add_usecase

        x_operands = [-1, 0, 1]
        y_operands = [-1, 0, 1]

        types_list = [(types.int32, types.int32),
                      (types.int64, types.int64)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

        x_operands = [0, 1]
        y_operands = [0, 1]

        types_list = [(types.byte, types.byte),
                      (types.uint32, types.uint32),
                      (types.uint64, types.uint64)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

    def test_add_ints_array(self, flags=enable_pyobj_flags):

        pyfunc = add_usecase

        array = np.arange(-10, 10, dtype=np.int32)

        x_operands = [array]
        y_operands = [array]

        arraytype = types.Array(types.int32, 1, 'C')
        types_list = [(arraytype, arraytype)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)
        
    def test_add_ints_npm(self):
        self.test_add_ints(flags=Noflags)

    @unittest.expectedFailure
    def test_add_ints_array_npm(self):
        self.test_add_ints_array(flags=Noflags)

    def test_add_floats(self, flags=enable_pyobj_flags):

        pyfunc = add_usecase

        x_operands = [-1.1, 0.0, 1.1]
        y_operands = [-1.1, 0.0, 1.1]

        types_list = [(types.float32, types.float32),
                      (types.float64, types.float64)]

        self.run_test_floats(pyfunc, x_operands, y_operands, types_list,
                             flags=flags)

    def test_add_floats_array(self, flags=enable_pyobj_flags):
        pyfunc = add_usecase
        array = np.arange(-1, 1, 0.1, dtype=np.float32)

        x_operands = [array]
        y_operands = [array]

        arraytype = types.Array(types.float32, 1, 'C')
        types_list = [(arraytype, arraytype)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

    def test_add_floats_npm(self):
        self.test_add_floats(flags=Noflags)


    @unittest.expectedFailure
    def test_add_floats_array_npm(self):
        self.test_add_floats_array(flags=Noflags)

    def test_sub_ints(self, flags=enable_pyobj_flags):

        pyfunc = sub_usecase

        x_operands = [-1, 0, 1]
        y_operands = [-1, 0, 1]

        types_list = [(types.int32, types.int32),
                      (types.int64, types.int64)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

        # Unsigned version will overflow and wraparound
        x_operands = [1, 2]
        y_operands = [0, 1]

        types_list = [(types.byte, types.byte),
                      (types.uint32, types.uint32),
                      (types.uint64, types.uint64)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

    def test_sub_ints_array(self, flags=enable_pyobj_flags):

        pyfunc = sub_usecase

        array = np.arange(-10, 10, dtype=np.int32)

        x_operands = [array]
        y_operands = [array]

        arraytype = types.Array(types.int32, 1, 'C')
        types_list = [(arraytype, arraytype)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

    def test_sub_ints_npm(self):
        self.test_sub_ints(flags=Noflags)

    @unittest.expectedFailure
    def test_sub_ints_array_npm(self):
        self.test_sub_ints_array(flags=Noflags)

    def test_sub_floats(self, flags=enable_pyobj_flags):

        pyfunc = sub_usecase

        x_operands = [-1.1, 0.0, 1.1]
        y_operands = [-1.1, 0.0, 1.1]

        types_list = [(types.float32, types.float32),
                      (types.float64, types.float64)]

        self.run_test_floats(pyfunc, x_operands, y_operands, types_list,
                             flags=flags)

    def test_sub_floats_array(self, flags=enable_pyobj_flags):

        pyfunc = sub_usecase

        array = np.arange(-1, 1, 0.1, dtype=np.float32)

        x_operands = [array]
        y_operands = [array]

        arraytype = types.Array(types.float32, 1, 'C')
        types_list = [(arraytype, arraytype)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

    def test_sub_floats_npm(self):
        self.test_sub_floats(flags=Noflags)

    @unittest.expectedFailure
    def test_sub_floats_array_npm(self):
        self.test_sub_floats_array(flags=Noflags)

    def test_mul_ints(self, flags=enable_pyobj_flags):

        pyfunc = mul_usecase

        x_operands = [-1, 0, 1]
        y_operands = [-1, 0, 1]

        types_list = [(types.int32, types.int32),
                      (types.int64, types.int64)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

        x_operands = [0, 1]
        y_operands = [0, 1]

        types_list = [(types.byte, types.byte),
                      (types.uint32, types.uint32),
                      (types.uint64, types.uint64)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

    def test_mul_ints_array(self, flags=enable_pyobj_flags):

        pyfunc = mul_usecase

        array = np.arange(-10, 10, dtype=np.int32)

        x_operands = [array]
        y_operands = [array]

        arraytype = types.Array(types.int32, 1, 'C')
        types_list = [(arraytype, arraytype)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

    def test_mul_ints_npm(self):
        self.test_mul_ints(flags=Noflags)

    @unittest.expectedFailure
    def test_mul_ints_array_npm(self):
        self.test_mul_ints_array(flags=Noflags)

    def test_mul_floats(self, flags=enable_pyobj_flags):

        pyfunc = mul_usecase

        x_operands = [-111.111, 0.0, 111.111]
        y_operands = [-111.111, 0.0, 111.111]

        types_list = [(types.float32, types.float32),
                      (types.float64, types.float64)]

        self.run_test_floats(pyfunc, x_operands, y_operands, types_list,
                             flags=flags)

    def test_mul_floats_array(self, flags=enable_pyobj_flags):

        pyfunc = mul_usecase
        array = np.arange(-1, 1, 0.1, dtype=np.float32)

        x_operands = [array]
        y_operands = [array]

        arraytype = types.Array(types.float32, 1, 'C')
        types_list = [(arraytype, arraytype)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

    def test_mul_floats_npm(self):
        self.test_mul_floats(flags=Noflags)

    @unittest.expectedFailure
    def test_mul_floats_array_npm(self):
        self.test_mul_floats_array(flags=Noflags)

    def test_div_ints(self, flags=enable_pyobj_flags):
        if PYVERSION >= (3, 0):
            # Due to true division returning float
            tester = self.run_test_floats
        else:
            tester = self.run_test_ints

        pyfunc = div_usecase

        x_operands = [-1, 0, 1, 2, 3]
        y_operands = [-3, -2, -1, 1]

        types_list = [(types.int32, types.int32),
                      (types.int64, types.int64)]

        tester(pyfunc, x_operands, y_operands, types_list, flags=flags)

        x_operands = [0, 1, 2, 3]
        y_operands = [1, 2, 3]

        types_list = [(types.byte, types.byte),
                      (types.uint32, types.uint32),
                      (types.uint64, types.uint64)]

        tester(pyfunc, x_operands, y_operands, types_list, flags=flags)

    def test_div_ints_array(self, flags=enable_pyobj_flags):
        pyfunc = div_usecase
        array = np.array([-10, -9, -2, -1, 1, 2, 9, 10], dtype=np.int32)

        x_operands = [array]
        y_operands = [array]

        arraytype = types.Array(types.int32, 1, 'C')
        types_list = [(arraytype, arraytype)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

    def test_div_ints_npm(self):
        self.test_div_ints(flags=Noflags)

    @unittest.expectedFailure
    def test_div_ints_array_npm(self):
        self.test_div_ints_array(flags=Noflags)

    def test_div_floats(self, flags=enable_pyobj_flags):

        pyfunc = div_usecase

        x_operands = [-111.111, 0.0, 2.2]
        y_operands = [-2.2, 1.0, 111.111]

        types_list = [(types.float32, types.float32),
                      (types.float64, types.float64)]

        self.run_test_floats(pyfunc, x_operands, y_operands, types_list,
                             flags=flags)

    def test_div_floats_array(self, flags=enable_pyobj_flags):

        pyfunc = div_usecase

        array = np.concatenate((np.arange(0.1, 1.1, 0.1, dtype=np.float32),
                               np.arange(-1.0, 0.0, 0.1, dtype=np.float32)))

        x_operands = [array]
        y_operands = [array]

        arraytype = types.Array(types.float32, 1, 'C')
        types_list = [(arraytype, arraytype)]

        self.run_test_floats(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

    def test_div_floats_npm(self):
        self.test_div_floats(flags=Noflags)

    @unittest.expectedFailure
    def test_div_floats_array_npm(self):
        self.test_div_floats_array(flags=Noflags)

    def test_truediv_ints(self, flags=enable_pyobj_flags):
        pyfunc = truediv_usecase

        x_operands = [0, 1, 2, 3]
        y_operands = [1, 1, 2, 3]

        types_list = [(types.uint32, types.uint32),
                      (types.uint64, types.uint64),
                      (types.int32, types.int32),
                      (types.int64, types.int64)]

        self.run_test_floats(pyfunc, x_operands, y_operands, types_list,
                             flags=flags)

    def test_truediv_ints_npm(self):
        self.test_truediv_ints(flags=Noflags)

    def test_truediv_floats(self, flags=enable_pyobj_flags):
        pyfunc = truediv_usecase

        x_operands = [-111.111, 0.0, 2.2]
        y_operands = [-2.2, 1.0, 111.111]

        types_list = [(types.float32, types.float32),
                      (types.float64, types.float64)]

        self.run_test_floats(pyfunc, x_operands, y_operands, types_list,
                             flags=flags)

    def test_truediv_floats_npm(self):
        self.test_truediv_floats(flags=Noflags)

    def test_floordiv_floats(self, flags=enable_pyobj_flags):
        pyfunc = floordiv_usecase

        x_operands = [-111.111, 0.0, 2.2]
        y_operands = [-2.2, 1.0, 111.111]

        types_list = [(types.float32, types.float32),
                      (types.float64, types.float64)]

        self.run_test_floats(pyfunc, x_operands, y_operands, types_list,
                             flags=flags)

    def test_floordiv_floats_npm(self):
        self.test_floordiv_floats(flags=Noflags)

    def test_mod_ints(self, flags=enable_pyobj_flags):

        pyfunc = mod_usecase

        x_operands = [-1, 0, 1, 2, 3]
        y_operands = [-3, -2, -1, 1]

        types_list = [(types.int32, types.int32),
                      (types.int64, types.int64)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

        x_operands = [0, 1, 2, 3]
        y_operands = [1, 2, 3]

        types_list = [(types.byte, types.byte),
                      (types.uint32, types.uint32),
                      (types.uint64, types.uint64)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

    def test_mod_ints_array(self, flags=enable_pyobj_flags):

        pyfunc = mod_usecase

        array = np.concatenate((np.arange(1, 11, dtype=np.int32),
                               np.arange(-10, 0, dtype=np.int32)))

        x_operands = [array]
        y_operands = [array]

        arraytype = types.Array(types.int32, 1, 'C')
        types_list = [(arraytype, arraytype)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

    def test_mod_ints_npm(self):
        self.test_mod_ints(flags=Noflags)

    @unittest.expectedFailure
    def test_mod_ints_array_npm(self):
        self.test_mod_ints_array(flags=Noflags)

    def test_mod_floats(self, flags=enable_pyobj_flags):

        pyfunc = mod_usecase

        x_operands = [-111.111, 0.0, 2.2]
        y_operands = [-2.2, 1.0, 111.111]

        types_list = [(types.float32, types.float32),
                      (types.float64, types.float64)]

        self.run_test_floats(pyfunc, x_operands, y_operands, types_list,
                             flags=flags)

    def test_mod_floats_array(self, flags=enable_pyobj_flags):

        pyfunc = mod_usecase

        array = np.arange(-1, 1, 0.1, dtype=np.float32)

        x_operands = [array]
        y_operands = [array]

        arraytype = types.Array(types.float32, 1, 'C')
        types_list = [(arraytype, arraytype)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

    def test_mod_floats_npm(self):
        self.test_mod_floats(flags=Noflags)

    @unittest.expectedFailure
    def test_mod_floats_array_npm(self):
        self.test_mod_floats_array(flags=Noflags)

    def test_pow_ints(self, flags=enable_pyobj_flags):

        pyfunc = pow_usecase

        x_operands = [-2, -1, 0, 1, 2]
        y_operands = [0, 1, 2]

        types_list = [(types.int32, types.int32),
                      (types.int64, types.int64)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

        x_operands = [0, 1, 2]
        y_operands = [0, 1, 2]

        types_list = [(types.byte, types.byte),
                      (types.uint32, types.uint32),
                      (types.uint64, types.uint64)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

    def test_pow_ints_array(self, flags=enable_pyobj_flags):

        pyfunc = pow_usecase

        array = np.arange(-10, 10, dtype=np.int32)

        x_operands = [array]
        y_operands = [array]

        arraytype = types.Array(types.int32, 1, 'C')
        types_list = [(arraytype, arraytype)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

    def test_pow_ints_npm(self):
        self.test_pow_ints(flags=Noflags)

    @unittest.expectedFailure
    def test_pow_ints_array_npm(self):
        self.test_pow_ints_array(flags=Noflags)

    def test_pow_floats(self, flags=enable_pyobj_flags):

        pyfunc = pow_usecase

        x_operands = [-222.222, -111.111, 111.111, 222.222]
        y_operands = [-2, -1, 0, 1, 2]

        types_list = [(types.float32, types.float32),
                      (types.float64, types.float64)]

        self.run_test_floats(pyfunc, x_operands, y_operands, types_list,
                             flags=flags)

        x_operands = [0.0]
        y_operands = [0, 1, 2]  # TODO native handling of 0 ** negative power

        types_list = [(types.float32, types.float32),
                      (types.float64, types.float64)]

        self.run_test_floats(pyfunc, x_operands, y_operands, types_list,
                             flags=flags)

    def test_pow_floats_array(self, flags=enable_pyobj_flags):

        pyfunc = pow_usecase
        # NOTE
        # If x is finite negative and y is finite but not an integer,
        # it causes a domain error
        array = np.arange(0.1, 1, 0.1, dtype=np.float32)

        x_operands = [array]
        y_operands = [array]

        arraytype = types.Array(types.float32, 1, 'C')
        types_list = [(arraytype, arraytype)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)


        x_array = np.arange(-1, 0.1, 0.1, dtype=np.float32)
        y_array = np.arange(len(x_array), dtype=np.float32)

        x_operands = [x_array]
        y_operands = [y_array]

        arraytype = types.Array(types.float32, 1, 'C')
        types_list = [(arraytype, arraytype)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

    def test_pow_floats_npm(self):
        self.test_pow_floats(flags=Noflags)

    @unittest.expectedFailure
    def test_pow_floats_array_npm(self):
        self.test_pow_floats_array(flags=Noflags)

    def test_add_complex(self, flags=enable_pyobj_flags):
        pyfunc = add_usecase

        x_operands = [1+0j, 1j, -1-1j]
        y_operands = x_operands

        types_list = [(types.complex64, types.complex64),
                      (types.complex128, types.complex128),]

        self.run_test_floats(pyfunc, x_operands, y_operands, types_list,
                             flags=flags)

    def test_add_complex_npm(self):
        self.test_add_complex(flags=Noflags)

    def test_sub_complex(self, flags=enable_pyobj_flags):
        pyfunc = sub_usecase

        x_operands = [1+0j, 1j, -1-1j]
        y_operands = [1, 2, 3]

        types_list = [(types.complex64, types.complex64),
                      (types.complex128, types.complex128),]

        self.run_test_floats(pyfunc, x_operands, y_operands, types_list,
                             flags=flags)

    def test_sub_complex_npm(self):
        self.test_sub_complex(flags=Noflags)

    def test_mul_complex(self, flags=enable_pyobj_flags):
        pyfunc = mul_usecase

        x_operands = [1+0j, 1j, -1-1j]
        y_operands = [1, 2, 3]

        types_list = [(types.complex64, types.complex64),
                      (types.complex128, types.complex128),]

        self.run_test_floats(pyfunc, x_operands, y_operands, types_list,
                             flags=flags)

    def test_mul_complex_npm(self):
        self.test_mul_complex(flags=Noflags)

    def test_div_complex(self, flags=enable_pyobj_flags):
        pyfunc = div_usecase

        x_operands = [1+0j, 1j, -1-1j]
        y_operands = [1, 2, 3]

        types_list = [(types.complex64, types.complex64),
                      (types.complex128, types.complex128),]

        self.run_test_floats(pyfunc, x_operands, y_operands, types_list,
                             flags=flags)

    def test_div_complex_npm(self):
        self.test_div_complex(flags=Noflags)

    def test_truediv_complex(self, flags=enable_pyobj_flags):
        pyfunc = truediv_usecase

        x_operands = [1+0j, 1j, -1-1j]
        y_operands = [1, 2, 3]

        types_list = [(types.complex64, types.complex64),
                      (types.complex128, types.complex128),]

        self.run_test_floats(pyfunc, x_operands, y_operands, types_list,
                             flags=flags)

    def test_truediv_complex_npm(self):
        self.test_truediv_complex(flags=Noflags)

    def test_mod_complex(self, flags=enable_pyobj_flags):
        pyfunc = mod_usecase

        try:
            cres = compile_isolated(pyfunc, (types.complex64, types.complex64))
        except typeinfer.TypingError as e:
            e.msg.startswith("Undeclared %(complex64, complex64)")
        else:
            self.fail("Complex % should trigger an undeclared error")

    def test_mod_complex_npm(self):
        self.test_mod_complex(flags=Noflags)

    def test_bitshift_left(self, flags=enable_pyobj_flags):

        pyfunc = bitshift_left_usecase

        x_operands = [0, 1]
        y_operands = [0, 1, 2, 4, 8, 16, 31]

        types_list = [(types.uint32, types.uint32)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

        x_operands = [0, 1]
        y_operands = [0, 1, 2, 4, 8, 16, 32, 63]

        types_list = [(types.uint64, types.uint64)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

        x_operands = [0, -1]
        y_operands = [0, 1, 2, 4, 8, 16, 31]

        types_list = [(types.int32, types.int32)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

        x_operands = [0, -1]
        y_operands = [0, 1, 2, 4, 8, 16, 32, 63]

        types_list = [(types.int64, types.int64)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

    def test_bitshift_left_npm(self):
        self.test_bitshift_left(flags=Noflags)

    def test_bitshift_right(self, flags=enable_pyobj_flags):

        pyfunc = bitshift_right_usecase

        x_operands = [0, 1, 2**32 - 1]
        y_operands = [0, 1, 2, 4, 8, 16, 31]

        types_list = [(types.uint32, types.uint32)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

        x_operands = [0, 1, 2**64 - 1]
        y_operands = [0, 1, 2, 4, 8, 16, 32, 63]

        types_list = [(types.uint64, types.uint64)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

        x_operands = [0, 1, -(2**31)]
        y_operands = [0, 1, 2, 4, 8, 16, 31]

        types_list = [(types.int32, types.int32)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

        x_operands = [0, -1, -(2**31)]
        y_operands = [0, 1, 2, 4, 8, 16, 32, 63]

        types_list = [(types.int64, types.int64)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

    def test_bitshift_right_npm(self):
        self.test_bitshift_right(flags=Noflags)

    def test_bitwise_and(self, flags=enable_pyobj_flags):

        pyfunc = bitwise_and_usecase

        x_operands = list(range(0, 8)) + [2**32 - 1]
        y_operands = list(range(0, 8)) + [2**32 - 1]

        types_list = [(types.uint32, types.uint32)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

        x_operands = list(range(0, 8)) + [2**64 - 1]
        y_operands = list(range(0, 8)) + [2**64 - 1]

        types_list = [(types.uint64, types.uint64)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

        x_operands = list(range(-4, 4)) + [-(2**31), 2**31 - 1]
        y_operands = list(range(-4, 4)) + [-(2**31), 2**31 - 1]

        types_list = [(types.int32, types.int32)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

        x_operands = list(range(-4, 4)) + [-(2**63), 2**63 - 1]
        y_operands = list(range(-4, 4)) + [-(2**63), 2**63 - 1]

        types_list = [(types.int64, types.int64)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

    def test_bitwise_and_npm(self):
        self.test_bitwise_and(flags=Noflags)

    def test_bitwise_or(self, flags=enable_pyobj_flags):

        pyfunc = bitwise_or_usecase

        x_operands = list(range(0, 8)) + [2**32 - 1]
        y_operands = list(range(0, 8)) + [2**32 - 1]

        types_list = [(types.uint32, types.uint32)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

        x_operands = list(range(0, 8)) + [2**64 - 1]
        y_operands = list(range(0, 8)) + [2**64 - 1]

        types_list = [(types.uint64, types.uint64)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

        x_operands = list(range(-4, 4)) + [-(2**31), 2**31 - 1]
        y_operands = list(range(-4, 4)) + [-(2**31), 2**31 - 1]

        types_list = [(types.int32, types.int32)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

        x_operands = list(range(-4, 4)) + [-(2**63), 2**63 - 1]
        y_operands = list(range(-4, 4)) + [-(2**63), 2**63 - 1]

        types_list = [(types.int64, types.int64)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

    def test_bitwise_or_npm(self):
        self.test_bitwise_or(flags=Noflags)

    def test_bitwise_xor(self, flags=enable_pyobj_flags):

        pyfunc = bitwise_xor_usecase

        x_operands = list(range(0, 8)) + [2**32 - 1]
        y_operands = list(range(0, 8)) + [2**32 - 1]

        types_list = [(types.uint32, types.uint32)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

        x_operands = list(range(0, 8)) + [2**64 - 1]
        y_operands = list(range(0, 8)) + [2**64 - 1]

        types_list = [(types.uint64, types.uint64)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

        x_operands = list(range(-4, 4)) + [-(2**31), 2**31 - 1]
        y_operands = list(range(-4, 4)) + [-(2**31), 2**31 - 1]

        types_list = [(types.int32, types.int32)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

        x_operands = list(range(-4, 4)) + [-(2**63), 2**63 - 1]
        y_operands = list(range(-4, 4)) + [-(2**63), 2**63 - 1]

        types_list = [(types.int64, types.int64)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

    def test_bitwise_xor_npm(self):
        self.test_bitwise_xor(flags=Noflags)

    def test_bitwise_not(self, flags=enable_pyobj_flags):

        pyfunc = bitwise_not_usecase

        x_operands = list(range(0, 8)) + [2**32 - 1]
        x_operands = [np.uint32(x) for x in x_operands]
        y_operands = [0]

        types_list = [(types.uint32, types.uint32)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

        x_operands = list(range(-4, 4)) + [-(2**31), 2**31 - 1]
        y_operands = [0]

        types_list = [(types.int32, types.int32)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

        x_operands = list(range(0, 8)) + [2**64 - 1]
        x_operands = [np.uint64(x) for x in x_operands]
        y_operands = [0]

        types_list = [(types.uint64, types.uint64)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

        x_operands = list(range(-4, 4)) + [-(2**63), 2**63 - 1]
        y_operands = [0]

        types_list = [(types.int64, types.int64)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

    def test_bitwise_not_npm(self):
        self.test_bitwise_not(flags=Noflags)

    def test_not(self):
        pyfunc = not_usecase

        values = [
            1,
            2,
            3,
            1.2,
            3.4j,
        ]

        cres = compile_isolated(pyfunc, (), flags=enable_pyobj_flags)
        cfunc = cres.entry_point
        for val in values:
            self.assertEqual(pyfunc(val), cfunc(val))

    def test_not_npm(self):
        pyfunc = not_usecase
        # test native mode
        argtys = [
            types.int8,
            types.int32,
            types.int64,
            types.float32,
            types.complex128,
        ]
        values = [
            1,
            2,
            3,
            1.2,
            3.4j,
        ]
        for ty, val in zip(argtys, values):
            cres = compile_isolated(pyfunc, [ty])
            self.assertEqual(cres.signature.return_type, types.boolean)
            cfunc = cres.entry_point
            self.assertEqual(pyfunc(val), cfunc(val))

    def test_negate_npm(self):
        pyfunc = negate_usecase
        # test native mode
        argtys = [
            types.int8,
            types.int32,
            types.int64,
            types.float32,
            types.float64,
            types.complex128,
        ]
        values = [
            1,
            2,
            3,
            1.2,
            2.4,
            3.4j,
        ]
        for ty, val in zip(argtys, values):
            cres = compile_isolated(pyfunc, [ty])
            cfunc = cres.entry_point
            self.assertAlmostEqual(pyfunc(val), cfunc(val))

    def test_negate(self):
        pyfunc = negate_usecase
        values = [
            1,
            2,
            3,
            1.2,
            3.4j,
        ]
        cres = compile_isolated(pyfunc, (), flags=enable_pyobj_flags)
        cfunc = cres.entry_point
        for val in values:
            self.assertEqual(pyfunc(val), cfunc(val))

    def test_unary_positive_npm(self):
        pyfunc = unary_positive_usecase
        # test native mode
        argtys = [
            types.int8,
            types.int32,
            types.int64,
            types.float32,
            types.float64,
            types.complex128,
        ]
        values = [
            1,
            2,
            3,
            1.2,
            2.4,
            3.4j,
        ]
        for ty, val in zip(argtys, values):
            cres = compile_isolated(pyfunc, [ty])
            cfunc = cres.entry_point
            self.assertAlmostEqual(pyfunc(val), cfunc(val))

    def test_unary_positive(self):
        pyfunc = unary_positive_usecase
        values = [
            1,
            2,
            3,
            1.2,
            3.4j,
        ]
        cres = compile_isolated(pyfunc, (), flags=enable_pyobj_flags)
        cfunc = cres.entry_point
        for val in values:
            self.assertEqual(pyfunc(val), cfunc(val))

    def test_sub_floats(self, flags=enable_pyobj_flags):

        pyfunc = sub_usecase

        x_operands = [-1.1, 0.0, 1.1]
        y_operands = [-1.1, 0.0, 1.1]

        types_list = [(types.float32, types.float32),
                      (types.float64, types.float64)]

        self.run_test_floats(pyfunc, x_operands, y_operands, types_list,
                             flags=flags)

    def test_sub_floats_array(self, flags=enable_pyobj_flags):

        pyfunc = sub_usecase

        array = np.arange(-1, 1, 0.1, dtype=np.float32)

        x_operands = [array]
        y_operands = [array]

        arraytype = types.Array(types.float32, 1, 'C')
        types_list = [(arraytype, arraytype)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)


if __name__ == '__main__':
    unittest.main()

