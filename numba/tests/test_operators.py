from __future__ import print_function

import copy
import itertools
import operator
import warnings

import numpy as np

from numba.compiler import compile_isolated, Flags
from numba import types, typeinfer
from numba.config import PYVERSION
from .support import TestCase
from numba.tests.true_div_usecase import truediv_usecase, itruediv_usecase
import numba.unittest_support as unittest

Noflags = Flags()

force_pyobj_flags = Flags()
force_pyobj_flags.set("enable_pyobject")

force_pyobj_flags = Flags()
force_pyobj_flags.set("force_pyobject")


class LiteralOperatorImpl(object):

    @staticmethod
    def add_usecase(x, y):
        return x + y

    @staticmethod
    def iadd_usecase(x, y):
        x += y
        return x

    @staticmethod
    def sub_usecase(x, y):
        return x - y

    @staticmethod
    def isub_usecase(x, y):
        x -= y
        return x

    @staticmethod
    def mul_usecase(x, y):
        return x * y

    @staticmethod
    def imul_usecase(x, y):
        x *= y
        return x

    @staticmethod
    def div_usecase(x, y):
        return x / y

    @staticmethod
    def idiv_usecase(x, y):
        x /= y
        return x

    @staticmethod
    def floordiv_usecase(x, y):
        return x // y

    @staticmethod
    def ifloordiv_usecase(x, y):
        x //= y
        return x

    truediv_usecase = staticmethod(truediv_usecase)
    itruediv_usecase = staticmethod(itruediv_usecase)

    @staticmethod
    def mod_usecase(x, y):
        return x % y

    @staticmethod
    def imod_usecase(x, y):
        x %= y
        return x

    @staticmethod
    def pow_usecase(x, y):
        return x ** y

    @staticmethod
    def ipow_usecase(x, y):
        x **= y
        return x

    @staticmethod
    def bitshift_left_usecase(x, y):
        return x << y

    @staticmethod
    def bitshift_ileft_usecase(x, y):
        x <<= y
        return x

    @staticmethod
    def bitshift_right_usecase(x, y):
        return x >> y

    @staticmethod
    def bitshift_iright_usecase(x, y):
        x >>= y
        return x

    @staticmethod
    def bitwise_and_usecase(x, y):
        return x & y

    @staticmethod
    def bitwise_iand_usecase(x, y):
        x &= y
        return x

    @staticmethod
    def bitwise_or_usecase(x, y):
        return x | y

    @staticmethod
    def bitwise_ior_usecase(x, y):
        x |= y
        return x

    @staticmethod
    def bitwise_xor_usecase(x, y):
        return x ^ y

    @staticmethod
    def bitwise_ixor_usecase(x, y):
        x ^= y
        return x

    @staticmethod
    def bitwise_not_usecase(x, y):
        return ~x

    @staticmethod
    def not_usecase(x):
        return not(x)

    @staticmethod
    def negate_usecase(x):
        return -x

    @staticmethod
    def unary_positive_usecase(x):
        return +x

    @staticmethod
    def lt_usecase(x, y):
        return x < y

    @staticmethod
    def le_usecase(x, y):
        return x <= y

    @staticmethod
    def gt_usecase(x, y):
        return x > y

    @staticmethod
    def ge_usecase(x, y):
        return x >= y

    @staticmethod
    def eq_usecase(x, y):
        return x == y

    @staticmethod
    def ne_usecase(x, y):
        return x != y


class FunctionalOperatorImpl(object):

    @staticmethod
    def add_usecase(x, y):
        return operator.add(x, y)

    @staticmethod
    def iadd_usecase(x, y):
        return operator.iadd(x, y)

    @staticmethod
    def sub_usecase(x, y):
        return operator.sub(x, y)

    @staticmethod
    def isub_usecase(x, y):
        return operator.isub(x, y)

    @staticmethod
    def mul_usecase(x, y):
        return operator.mul(x, y)

    @staticmethod
    def imul_usecase(x, y):
        return operator.imul(x, y)

    if PYVERSION >= (3, 0):
        div_usecase = NotImplemented
        idiv_usecase = NotImplemented
    else:
        @staticmethod
        def div_usecase(x, y):
            return operator.div(x, y)

        @staticmethod
        def idiv_usecase(x, y):
            return operator.idiv(x, y)

    @staticmethod
    def floordiv_usecase(x, y):
        return operator.floordiv(x, y)

    @staticmethod
    def ifloordiv_usecase(x, y):
        return operator.ifloordiv(x, y)

    @staticmethod
    def truediv_usecase(x, y):
        return operator.truediv(x, y)

    @staticmethod
    def itruediv_usecase(x, y):
        return operator.itruediv(x, y)

    @staticmethod
    def mod_usecase(x, y):
        return operator.mod(x, y)

    @staticmethod
    def imod_usecase(x, y):
        return operator.imod(x, y)

    @staticmethod
    def pow_usecase(x, y):
        return operator.pow(x, y)

    @staticmethod
    def ipow_usecase(x, y):
        return operator.ipow(x, y)

    @staticmethod
    def bitshift_left_usecase(x, y):
        return operator.lshift(x, y)

    @staticmethod
    def bitshift_ileft_usecase(x, y):
        return operator.ilshift(x, y)

    @staticmethod
    def bitshift_right_usecase(x, y):
        return operator.rshift(x, y)

    @staticmethod
    def bitshift_iright_usecase(x, y):
        return operator.irshift(x, y)

    @staticmethod
    def bitwise_and_usecase(x, y):
        return operator.and_(x, y)

    @staticmethod
    def bitwise_iand_usecase(x, y):
        return operator.iand(x, y)

    @staticmethod
    def bitwise_or_usecase(x, y):
        return operator.or_(x, y)

    @staticmethod
    def bitwise_ior_usecase(x, y):
        return operator.ior(x, y)

    @staticmethod
    def bitwise_xor_usecase(x, y):
        return operator.xor(x, y)

    @staticmethod
    def bitwise_ixor_usecase(x, y):
        return operator.ixor(x, y)

    @staticmethod
    def bitwise_not_usecase(x, y):
        return operator.invert(x)

    @staticmethod
    def not_usecase(x):
        return operator.not_(x)

    @staticmethod
    def negate_usecase(x):
        return operator.neg(x)

    @staticmethod
    def unary_positive_usecase(x):
        return operator.pos(x)

    @staticmethod
    def lt_usecase(x, y):
        return operator.lt(x, y)

    @staticmethod
    def le_usecase(x, y):
        return operator.le(x, y)

    @staticmethod
    def gt_usecase(x, y):
        return operator.gt(x, y)

    @staticmethod
    def ge_usecase(x, y):
        return operator.ge(x, y)

    @staticmethod
    def eq_usecase(x, y):
        return operator.eq(x, y)

    @staticmethod
    def ne_usecase(x, y):
        return operator.ne(x, y)


class TestOperators(TestCase):

    op = LiteralOperatorImpl

    def run_test_ints(self, pyfunc, x_operands, y_operands, types_list,
                      flags=force_pyobj_flags):
        if pyfunc is NotImplemented:
            self.skipTest("test irrelevant on this version of Python")
        for arg_types in types_list:
            cr = compile_isolated(pyfunc, arg_types, flags=flags)
            cfunc = cr.entry_point
            for x, y in itertools.product(x_operands, y_operands):
                # For inplace ops, we check that the first operand
                # was correctly mutated.
                x_got = copy.copy(x)
                x_expected = copy.copy(x)
                got = cfunc(x_got, y)
                expected = pyfunc(x_expected, y)
                self.assertTrue(np.all(got == expected),
                                "mismatch for (%r, %r): %r != %r"
                                % (x, y, got, expected))
                self.assertTrue(np.all(x_got == x_expected),
                                "mismatch for (%r, %r): %r != %r"
                                % (x, y, x_got, x_expected))

    def run_test_floats(self, pyfunc, x_operands, y_operands, types_list,
                        flags=force_pyobj_flags):
        if pyfunc is NotImplemented:
            self.skipTest("test irrelevant on this version of Python")
        for arg_types in types_list:
            cr = compile_isolated(pyfunc, arg_types, flags=flags)
            cfunc = cr.entry_point
            for x, y in itertools.product(x_operands, y_operands):
                # For inplace ops, we check that the first operand
                # was correctly mutated.
                x_got = copy.copy(x)
                x_expected = copy.copy(x)
                got = cfunc(x_got, y)
                expected = pyfunc(x_expected, y)
                self.assertTrue(np.allclose(got, expected))
                self.assertTrue(np.allclose(x_got, x_expected))

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

    def run_test_scalar_compare(self, pyfunc, flags=force_pyobj_flags,
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

    def run_test_array_compare(self, pyfunc, flags=force_pyobj_flags,
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

    #
    # Comparison operators
    #

    compare_scalar_operands = [-0.5, -1.0 + 1j, -1.0 + 2j, -0.5 + 1j, 1.5]
    compare_types = [types.int32, types.int64,
                     types.uint32, types.uint64,
                     types.float32, types.float64]
    compare_unordered_types = [types.complex64, types.complex128]

    def test_lt_scalar(self, flags=force_pyobj_flags):
        self.run_test_scalar_compare(self.op.lt_usecase, flags)

    def test_lt_scalar_npm(self):
        self.test_lt_scalar(flags=Noflags)

    def test_le_scalar(self, flags=force_pyobj_flags):
        self.run_test_scalar_compare(self.op.le_usecase, flags)

    def test_le_scalar_npm(self):
        self.test_le_scalar(flags=Noflags)

    def test_gt_scalar(self, flags=force_pyobj_flags):
        self.run_test_scalar_compare(self.op.gt_usecase, flags)

    def test_gt_scalar_npm(self):
        self.test_gt_scalar(flags=Noflags)

    def test_ge_scalar(self, flags=force_pyobj_flags):
        self.run_test_scalar_compare(self.op.ge_usecase, flags)

    def test_ge_scalar_npm(self):
        self.test_ge_scalar(flags=Noflags)

    def test_eq_scalar(self, flags=force_pyobj_flags):
        self.run_test_scalar_compare(self.op.eq_usecase, flags, ordered=False)

    def test_eq_scalar_npm(self):
        self.test_eq_scalar(flags=Noflags)

    def test_ne_scalar(self, flags=force_pyobj_flags):
        self.run_test_scalar_compare(self.op.ne_usecase, flags, ordered=False)

    def test_ne_scalar_npm(self):
        self.test_ne_scalar(flags=Noflags)

    def test_eq_array(self, flags=force_pyobj_flags):
        self.run_test_array_compare(self.op.eq_usecase, flags, ordered=False)

    def test_eq_array_npm(self):
        with self.assertTypingError():
            self.test_eq_array(flags=Noflags)

    def test_ne_array(self, flags=force_pyobj_flags):
        self.run_test_array_compare(self.op.ne_usecase, flags, ordered=False)

    def test_ne_array_npm(self):
        with self.assertTypingError():
            self.test_ne_array(flags=Noflags)

    def test_lt_array(self, flags=force_pyobj_flags):
        self.run_test_array_compare(self.op.lt_usecase, flags)

    def test_lt_array_npm(self):
        with self.assertTypingError():
            self.test_lt_array(flags=Noflags)

    def test_le_array(self, flags=force_pyobj_flags):
        self.run_test_array_compare(self.op.le_usecase, flags)

    def test_le_array_npm(self):
        with self.assertTypingError():
            self.test_le_array(flags=Noflags)

    def test_gt_array(self, flags=force_pyobj_flags):
        self.run_test_array_compare(self.op.gt_usecase, flags)

    def test_gt_array_npm(self):
        with self.assertTypingError():
            self.test_gt_array(flags=Noflags)

    def test_ge_array(self, flags=force_pyobj_flags):
        self.run_test_array_compare(self.op.ge_usecase, flags)

    def test_ge_array_npm(self):
        with self.assertTypingError():
            self.test_ge_array(flags=Noflags)

    #
    # Arithmetic operators
    #

    def run_binop_ints(self, pyfunc, flags=force_pyobj_flags):
        x_operands = [-2, 0, 1]
        y_operands = [-1, 1, 3]

        types_list = [(types.int32, types.int32),
                      (types.int64, types.int64)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

        x_operands = [2, 3]
        y_operands = [1, 2]

        types_list = [(types.byte, types.byte),
                      (types.uint32, types.uint32),
                      (types.uint64, types.uint64)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

    def run_binop_floats(self, pyfunc, flags=force_pyobj_flags):
        x_operands = [-1.1, 0.0, 1.1]
        y_operands = [-1.5, 0.8, 2.1]

        types_list = [(types.float32, types.float32),
                      (types.float64, types.float64)]

        self.run_test_floats(pyfunc, x_operands, y_operands, types_list,
                             flags=flags)

    def run_binop_complex(self, pyfunc, flags=force_pyobj_flags):
        x_operands = [-1.1 + 0.3j, 0.0 + 0.0j, 1.1j]
        y_operands = [-1.5 - 0.7j, 0.8j, 2.1 - 2.0j]

        types_list = [(types.complex64, types.complex64),
                      (types.complex128, types.complex128)]

        self.run_test_floats(pyfunc, x_operands, y_operands, types_list,
                             flags=flags)

    def run_binop_array_ints(self, pyfunc, flags=force_pyobj_flags):
        array = np.arange(-10, 10, dtype=np.int32)

        x_operands = [array[1:]]
        y_operands = [array[:-1]]

        arraytype = types.Array(types.int32, 1, 'C')
        types_list = [(arraytype, arraytype)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

    def run_binop_array_floats(self, pyfunc, flags=force_pyobj_flags):
        array = np.arange(-1, 1, 0.1, dtype=np.float32)

        x_operands = [array]
        y_operands = [array]

        arraytype = types.Array(types.float32, 1, 'C')
        types_list = [(arraytype, arraytype)]

        self.run_test_floats(pyfunc, x_operands, y_operands, types_list,
                             flags=flags)

    def run_binop_array_complex(self, pyfunc, flags=force_pyobj_flags):
        array = np.arange(-1, 1, 0.1, dtype=np.complex64) * (1.5 + 0.8j)

        x_operands = [array]
        y_operands = [array]

        arraytype = types.Array(types.complex64, 1, 'C')
        types_list = [(arraytype, arraytype)]

        self.run_test_floats(pyfunc, x_operands, y_operands, types_list,
                             flags=flags)

    def generate_binop_tests(ns, usecases, tp_runners, npm_array=False):
        for usecase in usecases:
            for tp_name, runner_name in tp_runners.items():
                for nopython in (False, True):
                    test_name = "test_%s_%s" % (usecase, tp_name)
                    if nopython:
                        test_name += "_npm"
                    flags = Noflags if nopython else force_pyobj_flags
                    usecase_name = "%s_usecase" % usecase

                    def inner(self, runner_name=runner_name,
                              usecase_name=usecase_name, flags=flags):
                        runner = getattr(self, runner_name)
                        op_usecase = getattr(self.op, usecase_name)
                        runner(op_usecase, flags)

                    if nopython and 'array' in tp_name and not npm_array:
                        def test_meth(self):
                            with self.assertTypingError():
                                inner()
                    else:
                        test_meth = inner

                    test_meth.__name__ = test_name
                    ns[test_name] = test_meth


    generate_binop_tests(locals(),
                         ('add', 'iadd', 'sub', 'isub', 'mul', 'imul'),
                         {'ints': 'run_binop_ints',
                          'floats': 'run_binop_floats',
                          'complex': 'run_binop_complex',
                          'ints_array': 'run_binop_array_ints',
                          'floats_array': 'run_binop_array_floats',
                          'complex_array': 'run_binop_array_complex',
                          })

    generate_binop_tests(locals(),
                         ('div', 'idiv', 'truediv', 'itruediv'),
                         {'ints': 'run_binop_ints',
                          'floats': 'run_binop_floats',
                          'complex': 'run_binop_complex',
                          'ints_array': 'run_binop_array_ints',
                          'floats_array': 'run_binop_array_floats',
                          'complex_array': 'run_binop_array_complex',
                          })

    # NOTE: floordiv and mod unsupported for complex numbers
    generate_binop_tests(locals(),
                         ('floordiv', 'ifloordiv', 'mod', 'imod'),
                         {'ints': 'run_binop_ints',
                          'floats': 'run_binop_floats',
                          'ints_array': 'run_binop_array_ints',
                          'floats_array': 'run_binop_array_floats',
                          })

    def run_pow_ints(self, pyfunc, flags=force_pyobj_flags):
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

    def run_pow_floats(self, pyfunc, flags=force_pyobj_flags):
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

    def run_pow_ints_array(self, pyfunc, flags=force_pyobj_flags):
        array = np.arange(-10, 10, dtype=np.int32)

        x_operands = [array]
        y_operands = [array]

        arraytype = types.Array(types.int32, 1, 'C')
        types_list = [(arraytype, arraytype)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

    def run_pow_floats_array(self, pyfunc, flags=force_pyobj_flags):
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

    # XXX power operator is unsupported on complex numbers (see issue #488)
    generate_binop_tests(locals(),
                         ('pow', 'ipow'),
                         {'ints': 'run_pow_ints',
                          'floats': 'run_pow_floats',
                          'ints_array': 'run_pow_ints_array',
                          'floats_array': 'run_pow_floats_array',
                          })

    def test_add_complex(self, flags=force_pyobj_flags):
        pyfunc = self.op.add_usecase

        x_operands = [1+0j, 1j, -1-1j]
        y_operands = x_operands

        types_list = [(types.complex64, types.complex64),
                      (types.complex128, types.complex128),]

        self.run_test_floats(pyfunc, x_operands, y_operands, types_list,
                             flags=flags)

    def test_add_complex_npm(self):
        self.test_add_complex(flags=Noflags)

    def test_sub_complex(self, flags=force_pyobj_flags):
        pyfunc = self.op.sub_usecase

        x_operands = [1+0j, 1j, -1-1j]
        y_operands = [1, 2, 3]

        types_list = [(types.complex64, types.complex64),
                      (types.complex128, types.complex128),]

        self.run_test_floats(pyfunc, x_operands, y_operands, types_list,
                             flags=flags)

    def test_sub_complex_npm(self):
        self.test_sub_complex(flags=Noflags)

    def test_mul_complex(self, flags=force_pyobj_flags):
        pyfunc = self.op.mul_usecase

        x_operands = [1+0j, 1j, -1-1j]
        y_operands = [1, 2, 3]

        types_list = [(types.complex64, types.complex64),
                      (types.complex128, types.complex128),]

        self.run_test_floats(pyfunc, x_operands, y_operands, types_list,
                             flags=flags)

    def test_mul_complex_npm(self):
        self.test_mul_complex(flags=Noflags)

    def test_div_complex(self, flags=force_pyobj_flags):
        pyfunc = self.op.div_usecase

        x_operands = [1+0j, 1j, -1-1j]
        y_operands = [1, 2, 3]

        types_list = [(types.complex64, types.complex64),
                      (types.complex128, types.complex128),]

        self.run_test_floats(pyfunc, x_operands, y_operands, types_list,
                             flags=flags)

    def test_div_complex_npm(self):
        self.test_div_complex(flags=Noflags)

    def test_truediv_complex(self, flags=force_pyobj_flags):
        pyfunc = self.op.truediv_usecase

        x_operands = [1+0j, 1j, -1-1j]
        y_operands = [1, 2, 3]

        types_list = [(types.complex64, types.complex64),
                      (types.complex128, types.complex128),]

        self.run_test_floats(pyfunc, x_operands, y_operands, types_list,
                             flags=flags)

    def test_truediv_complex_npm(self):
        self.test_truediv_complex(flags=Noflags)

    def test_mod_complex(self, flags=force_pyobj_flags):
        pyfunc = self.op.mod_usecase
        with self.assertTypingError():
            cres = compile_isolated(pyfunc, (types.complex64, types.complex64))

    def test_mod_complex_npm(self):
        self.test_mod_complex(flags=Noflags)

    #
    # Bitwise operators
    #

    def run_bitshift_left(self, pyfunc, flags=force_pyobj_flags):
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

    def run_bitshift_array(self, pyfunc, flags=force_pyobj_flags):
        array = np.arange(0, 10, dtype=np.int32)

        x_operands = [array[:-1]]
        y_operands = [array[1:]]

        arraytype = types.Array(types.int32, 1, 'C')
        types_list = [(arraytype, arraytype)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

    generate_binop_tests(locals(),
                         ('bitshift_left', 'bitshift_ileft'),
                         {'ints': 'run_bitshift_left',
                          'ints_array': 'run_bitshift_array',
                          })

    def run_bitshift_right(self, pyfunc, flags=force_pyobj_flags):
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

    generate_binop_tests(locals(),
                         ('bitshift_right', 'bitshift_iright'),
                         {'ints': 'run_bitshift_right',
                          'ints_array': 'run_bitshift_array',
                          })

    def run_logical(self, pyfunc, flags=force_pyobj_flags):
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

    def run_logical_array(self, pyfunc, flags=force_pyobj_flags):
        dtype = np.int32
        array = np.concatenate([
            np.array([-(2**31), 2**31 - 1], dtype=dtype),
            np.arange(-10, 10, dtype=dtype),
            ])

        x_operands = [array[:-1]]
        y_operands = [array[1:]]

        arraytype = types.Array(types.int32, 1, 'C')
        types_list = [(arraytype, arraytype)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

    generate_binop_tests(locals(),
                         ('bitwise_and', 'bitwise_iand',
                          'bitwise_or', 'bitwise_ior',
                          'bitwise_xor', 'bitwise_ixor'),
                         {'ints': 'run_logical',
                          'ints_array': 'run_logical_array',
                          })

    #
    # Unary operators
    #

    def test_bitwise_not(self, flags=force_pyobj_flags):

        pyfunc = self.op.bitwise_not_usecase

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
        pyfunc = self.op.not_usecase

        values = [
            1,
            2,
            3,
            1.2,
            3.4j,
        ]

        cres = compile_isolated(pyfunc, (), flags=force_pyobj_flags)
        cfunc = cres.entry_point
        for val in values:
            self.assertEqual(pyfunc(val), cfunc(val))

    def test_not_npm(self):
        pyfunc = self.op.not_usecase
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

    # XXX test_negate should check for negative and positive zeros and infinites

    def test_negate_npm(self):
        pyfunc = self.op.negate_usecase
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
        pyfunc = self.op.negate_usecase
        values = [
            1,
            2,
            3,
            1.2,
            3.4j,
        ]
        cres = compile_isolated(pyfunc, (), flags=force_pyobj_flags)
        cfunc = cres.entry_point
        for val in values:
            self.assertEqual(pyfunc(val), cfunc(val))

    def test_unary_positive_npm(self):
        pyfunc = self.op.unary_positive_usecase
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
        pyfunc = self.op.unary_positive_usecase
        values = [
            1,
            2,
            3,
            1.2,
            3.4j,
        ]
        cres = compile_isolated(pyfunc, (), flags=force_pyobj_flags)
        cfunc = cres.entry_point
        for val in values:
            self.assertEqual(pyfunc(val), cfunc(val))


class TestOperatorModule(TestOperators):

    op = FunctionalOperatorImpl


if __name__ == '__main__':
    unittest.main()

