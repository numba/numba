from __future__ import print_function

import copy
import itertools
import operator
import warnings

import numpy as np

import numba.unittest_support as unittest
from numba.compiler import compile_isolated, Flags
from numba import types, typeinfer, utils, errors
from numba.config import PYVERSION
from .support import TestCase, tag
from .true_div_usecase import truediv_usecase, itruediv_usecase
from .matmul_usecase import (matmul_usecase, imatmul_usecase, DumbMatrix,
                             needs_matmul, needs_blas)

Noflags = Flags()

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
    if matmul_usecase:
        matmul_usecase = staticmethod(matmul_usecase)
        imatmul_usecase = staticmethod(imatmul_usecase)

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
    def bitwise_not_usecase_binary(x, _unused):
        return ~x

    @staticmethod
    def bitwise_not_usecase(x):
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

    @staticmethod
    def in_usecase(x, y):
        return x in y

    @staticmethod
    def not_in_usecase(x, y):
        return x not in y


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
    def matmul_usecase(x, y):
        return operator.matmul(x, y)

    @staticmethod
    def imatmul_usecase(x, y):
        return operator.imatmul(x, y)

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
    def bitwise_not_usecase_binary(x, _unused):
        return operator.invert(x)

    @staticmethod
    def bitwise_not_usecase(x):
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

    @staticmethod
    def in_usecase(x, y):
        return operator.contains(y, x)

    @staticmethod
    def not_in_usecase(x, y):
        return not operator.contains(y, x)


class TestOperators(TestCase):
    """
    Test standard Python operators on scalars.

    NOTE: operators on array are generally tested in test_ufuncs.
    """

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
                self.assertPreciseEqual(
                    got, expected,
                    msg="mismatch for (%r, %r) with types %s: %r != %r"
                        % (x, y, arg_types, got, expected))
                self.assertPreciseEqual(
                    x_got, x_expected,
                    msg="mismatch for (%r, %r) with types %s: %r != %r"
                        % (x, y, arg_types, x_got, x_expected))

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

    @tag('important')
    def test_lt_scalar_npm(self):
        self.test_lt_scalar(flags=Noflags)

    def test_le_scalar(self, flags=force_pyobj_flags):
        self.run_test_scalar_compare(self.op.le_usecase, flags)

    @tag('important')
    def test_le_scalar_npm(self):
        self.test_le_scalar(flags=Noflags)

    def test_gt_scalar(self, flags=force_pyobj_flags):
        self.run_test_scalar_compare(self.op.gt_usecase, flags)

    @tag('important')
    def test_gt_scalar_npm(self):
        self.test_gt_scalar(flags=Noflags)

    def test_ge_scalar(self, flags=force_pyobj_flags):
        self.run_test_scalar_compare(self.op.ge_usecase, flags)

    @tag('important')
    def test_ge_scalar_npm(self):
        self.test_ge_scalar(flags=Noflags)

    def test_eq_scalar(self, flags=force_pyobj_flags):
        self.run_test_scalar_compare(self.op.eq_usecase, flags, ordered=False)

    @tag('important')
    def test_eq_scalar_npm(self):
        self.test_eq_scalar(flags=Noflags)

    def test_ne_scalar(self, flags=force_pyobj_flags):
        self.run_test_scalar_compare(self.op.ne_usecase, flags, ordered=False)

    @tag('important')
    def test_ne_scalar_npm(self):
        self.test_ne_scalar(flags=Noflags)


    #
    # Arithmetic operators
    #

    def run_binop_bools(self, pyfunc, flags=force_pyobj_flags):
        x_operands = [False, False, True, True]
        y_operands = [False, True, False, True]

        types_list = [(types.boolean, types.boolean)]

        self.run_test_ints(pyfunc, x_operands, y_operands, types_list,
                           flags=flags)

    def run_binop_ints(self, pyfunc, flags=force_pyobj_flags):
        x_operands = [-5, 0, 1, 2]
        y_operands = [-3, -1, 1, 3]

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

                    if nopython:
                        test_meth = tag('important')(test_meth)

                    ns[test_name] = test_meth


    generate_binop_tests(locals(),
                         ('add', 'iadd', 'sub', 'isub', 'mul', 'imul'),
                         {'ints': 'run_binop_ints',
                          'floats': 'run_binop_floats',
                          'complex': 'run_binop_complex',
                          })

    generate_binop_tests(locals(),
                         ('div', 'idiv', 'truediv', 'itruediv'),
                         {'ints': 'run_binop_ints',
                          'floats': 'run_binop_floats',
                          'complex': 'run_binop_complex',
                          })

    # NOTE: floordiv and mod unsupported for complex numbers
    generate_binop_tests(locals(),
                         ('floordiv', 'ifloordiv', 'mod', 'imod'),
                         {'ints': 'run_binop_ints',
                          'floats': 'run_binop_floats',
                          })

    def check_div_errors(self, usecase_name, msg, flags=force_pyobj_flags,
                         allow_complex=False):
        pyfunc = getattr(self.op, usecase_name)
        if pyfunc is NotImplemented:
            self.skipTest("%r not implemented" % (usecase_name,))
        # Signed and unsigned division can take different code paths,
        # test them both.
        arg_types = [types.int32, types.uint32, types.float64]
        if allow_complex:
            arg_types.append(types.complex128)
        for tp in arg_types:
            cr = compile_isolated(pyfunc, (tp, tp), flags=flags)
            cfunc = cr.entry_point
            with self.assertRaises(ZeroDivisionError) as cm:
                cfunc(1, 0)
            # Test exception message if not in object mode
            if flags is not force_pyobj_flags:
                self.assertIn(msg, str(cm.exception))

    def test_truediv_errors(self, flags=force_pyobj_flags):
        self.check_div_errors("truediv_usecase", "division by zero", flags=flags,
                              allow_complex=True)

    def test_truediv_errors_npm(self):
        self.test_truediv_errors(flags=Noflags)

    def test_floordiv_errors(self, flags=force_pyobj_flags):
        self.check_div_errors("floordiv_usecase", "division by zero", flags=flags)

    def test_floordiv_errors_npm(self):
        self.test_floordiv_errors(flags=Noflags)

    def test_div_errors(self, flags=force_pyobj_flags):
        self.check_div_errors("div_usecase", "division by zero", flags=flags)

    def test_div_errors_npm(self):
        self.test_div_errors(flags=Noflags)

    def test_mod_errors(self, flags=force_pyobj_flags):
        self.check_div_errors("mod_usecase", "modulo by zero", flags=flags)

    def test_mod_errors_npm(self):
        self.test_mod_errors(flags=Noflags)

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

    # XXX power operator is unsupported on complex numbers (see issue #488)
    generate_binop_tests(locals(),
                         ('pow', 'ipow'),
                         {'ints': 'run_pow_ints',
                          'floats': 'run_pow_floats',
                          })

    def test_add_complex(self, flags=force_pyobj_flags):
        pyfunc = self.op.add_usecase

        x_operands = [1+0j, 1j, -1-1j]
        y_operands = x_operands

        types_list = [(types.complex64, types.complex64),
                      (types.complex128, types.complex128),]

        self.run_test_floats(pyfunc, x_operands, y_operands, types_list,
                             flags=flags)

    @tag('important')
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

    @tag('important')
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

    @tag('important')
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

    @tag('important')
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

    @tag('important')
    def test_truediv_complex_npm(self):
        self.test_truediv_complex(flags=Noflags)

    def test_mod_complex(self, flags=force_pyobj_flags):
        pyfunc = self.op.mod_usecase
        with self.assertTypingError():
            cres = compile_isolated(pyfunc, (types.complex64, types.complex64))

    @tag('important')
    def test_mod_complex_npm(self):
        self.test_mod_complex(flags=Noflags)

    #
    # Matrix multiplication
    # (just check with simple values; computational tests are in test_linalg)
    #

    @needs_matmul
    def check_matmul_objmode(self, pyfunc, inplace):
        # Use dummy objects, to work with any Numpy / Scipy version
        # (and because Numpy 1.10 doesn't implement "@=")
        cres = compile_isolated(pyfunc, (), flags=force_pyobj_flags)
        cfunc = cres.entry_point
        a = DumbMatrix(3)
        b = DumbMatrix(4)
        got = cfunc(a, b)
        self.assertEqual(got.value, 12)
        if inplace:
            self.assertIs(got, a)
        else:
            self.assertIsNot(got, a)
            self.assertIsNot(got, b)

    @needs_matmul
    def test_matmul(self):
        self.check_matmul_objmode(self.op.matmul_usecase, inplace=False)

    @needs_matmul
    def test_imatmul(self):
        self.check_matmul_objmode(self.op.imatmul_usecase, inplace=True)

    @needs_blas
    @needs_matmul
    def check_matmul_npm(self, pyfunc):
        arrty = types.Array(types.float32, 1, 'C')
        cres = compile_isolated(pyfunc, (arrty, arrty), flags=Noflags)
        cfunc = cres.entry_point
        a = np.float32([1, 2])
        b = np.float32([3, 4])
        got = cfunc(a, b)
        self.assertPreciseEqual(got, np.dot(a, b))
        # Never inplace
        self.assertIsNot(got, a)
        self.assertIsNot(got, b)

    @tag('important')
    @needs_matmul
    def test_matmul_npm(self):
        self.check_matmul_npm(self.op.matmul_usecase)

    @tag('important')
    @needs_matmul
    def test_imatmul_npm(self):
        with self.assertTypingError() as raises:
            self.check_matmul_npm(self.op.imatmul_usecase)

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

    generate_binop_tests(locals(),
                         ('bitshift_left', 'bitshift_ileft'),
                         {'ints': 'run_bitshift_left',
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

    generate_binop_tests(locals(),
                         ('bitwise_and', 'bitwise_iand',
                          'bitwise_or', 'bitwise_ior',
                          'bitwise_xor', 'bitwise_ixor'),
                         {'ints': 'run_logical',
                          'bools': 'run_binop_bools',
                          })

    #
    # Unary operators
    #

    def test_bitwise_not(self, flags=force_pyobj_flags):
        pyfunc = self.op.bitwise_not_usecase_binary

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

        # For booleans, we follow Numpy semantics (i.e. ~True == False,
        # not ~True == -2)
        values = [False, False, True, True]
        values = list(map(np.bool_, values))

        pyfunc = self.op.bitwise_not_usecase
        cres = compile_isolated(pyfunc, (types.boolean,), flags=flags)
        cfunc = cres.entry_point
        for val in values:
            self.assertPreciseEqual(pyfunc(val), cfunc(val))

    @tag('important')
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

    @tag('important')
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

    @tag('important')
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

    def _check_in(self, pyfunc, flags):
        dtype = types.int64
        cres = compile_isolated(pyfunc, (dtype, types.UniTuple(dtype, 3)),
                                flags=flags)
        cfunc = cres.entry_point
        for i in (3, 4, 5, 6, 42):
            tup = (3, 42, 5)
            self.assertPreciseEqual(pyfunc(i, tup), cfunc(i, tup))

    def test_in(self, flags=force_pyobj_flags):
        self._check_in(self.op.in_usecase, flags)

    def test_in_npm(self):
        self.test_in(flags=Noflags)

    def test_not_in(self, flags=force_pyobj_flags):
        self._check_in(self.op.not_in_usecase, flags)

    def test_not_in_npm(self):
        self.test_not_in(flags=Noflags)


class TestOperatorModule(TestOperators):

    op = FunctionalOperatorImpl


class TestMixedInts(TestCase):
    """
    Tests for operator calls with mixed integer types.
    """

    op = LiteralOperatorImpl

    int_samples = [0, 1, 3, 10, 42, 127, 10000, -1, -3, -10, -42, -127, -10000]

    int_types = [types.int8, types.uint8, types.int64, types.uint64]
    signed_types = [tp for tp in int_types if tp.signed]
    unsigned_types = [tp for tp in int_types if not tp.signed]
    type_pairs = list(itertools.product(int_types, int_types))
    signed_pairs = [(u, v) for u, v in type_pairs
                    if u.signed or v.signed]
    unsigned_pairs = [(u, v) for u, v in type_pairs
                      if not (u.signed or v.signed)]

    def get_numpy_signed_upcast(self, *vals):
        bitwidth = max(v.dtype.itemsize * 8 for v in vals)
        bitwidth = max(bitwidth, types.intp.bitwidth)
        return getattr(np, "int%d" % bitwidth)

    def get_numpy_unsigned_upcast(self, *vals):
        bitwidth = max(v.dtype.itemsize * 8 for v in vals)
        bitwidth = max(bitwidth, types.intp.bitwidth)
        return getattr(np, "uint%d" % bitwidth)

    def get_typed_int(self, typ, val):
        return getattr(np, typ.name)(val)

    def get_control_signed(self, opname):
        op = getattr(operator, opname)
        def control_signed(a, b):
            tp = self.get_numpy_signed_upcast(a, b)
            return op(tp(a), tp(b))
        return control_signed

    def get_control_unsigned(self, opname):
        op = getattr(operator, opname)
        def control_unsigned(a, b):
            tp = self.get_numpy_unsigned_upcast(a, b)
            return op(tp(a), tp(b))
        return control_unsigned

    def run_binary(self, pyfunc, control_func, operands, types,
                   expected_type=utils.INT_TYPES, **assertPreciseEqualArgs):
        if pyfunc is NotImplemented:
            self.skipTest("test irrelevant on this version of Python")

        for xt, yt in types:
            cr = compile_isolated(pyfunc, (xt, yt), flags=Noflags)
            cfunc = cr.entry_point
            for x, y in itertools.product(operands, operands):
                # Get Numpy typed scalars for the given types and values
                x = self.get_typed_int(xt, x)
                y = self.get_typed_int(yt, y)
                expected = control_func(x, y)
                got = cfunc(x, y)
                self.assertIsInstance(got, expected_type)
                msg = ("mismatch for (%r, %r) with types %s"
                       % (x, y, (xt, yt)))
                self.assertPreciseEqual(got, expected, msg=msg,
                                        **assertPreciseEqualArgs)

    def run_unary(self, pyfunc, control_func, operands, types,
                  expected_type=utils.INT_TYPES):
        if pyfunc is NotImplemented:
            self.skipTest("test irrelevant on this version of Python")

        for xt in types:
            cr = compile_isolated(pyfunc, (xt,), flags=Noflags)
            cfunc = cr.entry_point
            for x in operands:
                x = self.get_typed_int(xt, x)
                expected = control_func(x)
                got = cfunc(x)
                self.assertIsInstance(got, expected_type)
                self.assertTrue(np.all(got == expected),
                                "mismatch for %r with type %s: %r != %r"
                                % (x, xt, got, expected))

    def run_arith_binop(self, pyfunc, opname, samples,
                        expected_type=utils.INT_TYPES):
        self.run_binary(pyfunc, self.get_control_signed(opname),
                        samples, self.signed_pairs, expected_type)
        self.run_binary(pyfunc, self.get_control_unsigned(opname),
                        samples, self.unsigned_pairs, expected_type)

    @tag('important')
    def test_add(self):
        self.run_arith_binop(self.op.add_usecase, 'add', self.int_samples)

    @tag('important')
    def test_sub(self):
        self.run_arith_binop(self.op.sub_usecase, 'sub', self.int_samples)

    @tag('important')
    def test_mul(self):
        self.run_arith_binop(self.op.mul_usecase, 'mul', self.int_samples)

    def test_floordiv(self):
        samples = [x for x in self.int_samples if x != 0]
        self.run_arith_binop(self.op.floordiv_usecase, 'floordiv', samples)

    def test_mod(self):
        samples = [x for x in self.int_samples if x != 0]
        self.run_arith_binop(self.op.mod_usecase, 'mod', samples)

    def test_pow(self):
        pyfunc = self.op.pow_usecase
        # Only test with positive values, as otherwise trying to write the
        # control function in terms of Python or Numpy power turns out insane.
        samples = [x for x in self.int_samples if x >= 0]
        self.run_arith_binop(pyfunc, 'pow', samples)

        # Now test all non-zero values, but only with signed types
        def control_signed(a, b):
            tp = self.get_numpy_signed_upcast(a, b)
            if b >= 0:
                return tp(a) ** tp(b)
            else:
                inv = tp(a) ** tp(-b)
                if inv == 0:
                    # Overflow
                    return 0
                return np.intp(1.0 / inv)
        samples = [x for x in self.int_samples if x != 0]
        signed_pairs = [(u, v) for u, v in self.type_pairs
                        if u.signed and v.signed]
        self.run_binary(pyfunc, control_signed,
                        samples, signed_pairs)

    def test_truediv(self):
        def control(a, b):
            return truediv_usecase(float(a), float(b))
        samples = [x for x in self.int_samples if x != 0]
        pyfunc = self.op.truediv_usecase

        # Note: there can be precision issues on x87
        # e.g. for `1 / 18446744073709541616`
        # -> 0x1.0000000000002p-64 vs. 0x1.0000000000003p-64.
        self.run_binary(pyfunc, control, samples, self.signed_pairs,
                        expected_type=float, prec='double')
        self.run_binary(pyfunc, control, samples, self.unsigned_pairs,
                        expected_type=float, prec='double')

    def test_and(self):
        self.run_arith_binop(self.op.bitwise_and_usecase, 'and_', self.int_samples)

    def test_or(self):
        self.run_arith_binop(self.op.bitwise_or_usecase, 'or_', self.int_samples)

    def test_xor(self):
        self.run_arith_binop(self.op.bitwise_xor_usecase, 'xor', self.int_samples)

    def run_shift_binop(self, pyfunc, opname):
        opfunc = getattr(operator, opname)
        def control_signed(a, b):
            tp = self.get_numpy_signed_upcast(a, b)
            return opfunc(tp(a), tp(b))
        def control_unsigned(a, b):
            tp = self.get_numpy_unsigned_upcast(a, b)
            return opfunc(tp(a), tp(b))

        samples = self.int_samples

        def check(xt, yt, control_func):
            cr = compile_isolated(pyfunc, (xt, yt), flags=Noflags)
            cfunc = cr.entry_point
            for x in samples:
                # Avoid shifting by more than the shiftand's bitwidth, as
                # we would hit undefined behaviour.
                maxshift = xt.bitwidth - 1
                for y in (0, 1, 3, 5, maxshift - 1, maxshift):
                    # Get Numpy typed scalars for the given types and values
                    x = self.get_typed_int(xt, x)
                    y = self.get_typed_int(yt, y)
                    expected = control_func(x, y)
                    got = cfunc(x, y)
                    msg = ("mismatch for (%r, %r) with types %s"
                           % (x, y, (xt, yt)))
                    self.assertPreciseEqual(got, expected, msg=msg)

        for xt, yt in self.signed_pairs:
            check(xt, yt, control_signed)
        for xt, yt in self.unsigned_pairs:
            check(xt, yt, control_unsigned)

    def test_lshift(self):
        self.run_shift_binop(self.op.bitshift_left_usecase, 'lshift')

    def test_rshift(self):
        self.run_shift_binop(self.op.bitshift_right_usecase, 'rshift')

    def test_unary_positive(self):
        def control(a):
            return a
        samples = self.int_samples
        pyfunc = self.op.unary_positive_usecase

        self.run_unary(pyfunc, control, samples, self.int_types)

    def test_unary_negative(self):
        def control_signed(a):
            tp = self.get_numpy_signed_upcast(a)
            return tp(-a)
        def control_unsigned(a):
            tp = self.get_numpy_unsigned_upcast(a)
            return tp(-a)
        samples = self.int_samples
        pyfunc = self.op.negate_usecase

        self.run_unary(pyfunc, control_signed, samples, self.signed_types)
        self.run_unary(pyfunc, control_unsigned, samples, self.unsigned_types)

    def test_invert(self):
        def control_signed(a):
            tp = self.get_numpy_signed_upcast(a)
            return tp(~a)
        def control_unsigned(a):
            tp = self.get_numpy_unsigned_upcast(a)
            return tp(~a)
        samples = self.int_samples
        pyfunc = self.op.bitwise_not_usecase

        self.run_unary(pyfunc, control_signed, samples, self.signed_types)
        self.run_unary(pyfunc, control_unsigned, samples, self.unsigned_types)


class TestMixedIntsOperatorModule(TestMixedInts):

    op = FunctionalOperatorImpl


if __name__ == '__main__':
    unittest.main()
