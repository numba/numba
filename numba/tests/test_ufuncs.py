from __future__ import print_function

import itertools
import sys
import warnings

import numpy as np
import functools

import numba.unittest_support as unittest
from numba.compiler import compile_isolated, Flags
from numba import types, utils
from numba import vectorize
from numba.config import PYVERSION
from numba.typeinfer import TypingError
from numba.tests.support import TestCase, CompilationCache


is32bits = tuple.__itemsize__ == 4
iswindows = sys.platform.startswith('win32')

enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

no_pyobj_flags = Flags()

def _unimplemented(func):
    """An 'expectedFailure' like decorator that only expects compilation errors
    caused by unimplemented functions that fail in no-python mode"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except TypingError:
            raise unittest._ExpectedFailure(sys.exc_info())
        raise unittest._UnexpectedSuccess

def _make_unary_ufunc_usecase(ufunc_name):
    ldict = {}
    exec("def fn(x,out):\n    np.{0}(x,out)".format(ufunc_name), globals(), ldict)
    fn = ldict["fn"]
    fn.__name__ = "{0}_usecase".format(ufunc_name)
    return fn


def _make_binary_ufunc_usecase(ufunc_name):
    ldict = {}
    exec("def fn(x,y,out):\n    np.{0}(x,y,out)".format(ufunc_name), globals(), ldict);
    fn = ldict['fn']
    fn.__name__ = "{0}_usecase".format(ufunc_name)
    return fn


def _as_dtype_value(tyargs, args):
    """Convert python values into numpy scalar objects.
    """
    return [np.dtype(str(ty)).type(val) for ty, val in zip(tyargs, args)]


class TestUFuncs(TestCase):

    def setUp(self):
        self.inputs = [
            (0, types.uint32),
            (1, types.uint32),
            (-1, types.int32),
            (0, types.int32),
            (1, types.int32),
            (0, types.uint64),
            (1, types.uint64),
            (-1, types.int64),
            (0, types.int64),
            (1, types.int64),

            (-0.5, types.float32),
            (0.0, types.float32),
            (0.5, types.float32),

            (-0.5, types.float64),
            (0.0, types.float64),
            (0.5, types.float64),

            (np.array([0,1], dtype='u4'), types.Array(types.uint32, 1, 'C')),
            (np.array([0,1], dtype='u8'), types.Array(types.uint64, 1, 'C')),
            (np.array([-1,0,1], dtype='i4'), types.Array(types.int32, 1, 'C')),
            (np.array([-1,0,1], dtype='i8'), types.Array(types.int64, 1, 'C')),
            (np.array([-0.5, 0.0, 0.5], dtype='f4'), types.Array(types.float32, 1, 'C')),
            (np.array([-0.5, 0.0, 0.5], dtype='f8'), types.Array(types.float64, 1, 'C')),
            ]
        self.cache = CompilationCache()

    def unary_ufunc_test(self, ufunc_name, flags=enable_pyobj_flags,
                         skip_inputs=[], additional_inputs=[],
                         int_output_type=None, float_output_type=None):
        ufunc = _make_unary_ufunc_usecase(ufunc_name)

        inputs = list(self.inputs)
        inputs.extend(additional_inputs)

        pyfunc = ufunc

        for input_tuple in inputs:
            input_operand = input_tuple[0]
            input_type = input_tuple[1]

            if input_type in skip_inputs:
                continue

            ty = input_type
            if isinstance(ty, types.Array):
                ty = ty.dtype

            if ty in types.signed_domain:
                if int_output_type:
                    output_type = types.Array(int_output_type, 1, 'C')
                else:
                    output_type = types.Array(types.int64, 1, 'C')
            elif ty in types.unsigned_domain:
                if int_output_type:
                    output_type = types.Array(int_output_type, 1, 'C')
                else:
                    output_type = types.Array(types.uint64, 1, 'C')
            else:
                if float_output_type:
                    output_type = types.Array(float_output_type, 1, 'C')
                else:
                    output_type = types.Array(types.float64, 1, 'C')

            cr = self.cache.compile(pyfunc, (input_type, output_type),
                                    flags=flags)
            cfunc = cr.entry_point

            if isinstance(input_operand, np.ndarray):
                result = np.zeros(input_operand.size,
                                  dtype=output_type.dtype.name)
                expected = np.zeros(input_operand.size,
                                    dtype=output_type.dtype.name)
            else:
                result = np.zeros(1, dtype=output_type.dtype.name)
                expected = np.zeros(1, dtype=output_type.dtype.name)

            invalid_flag = False
            with warnings.catch_warnings(record=True) as warnlist:
                warnings.simplefilter('always')

                pyfunc(input_operand, expected)

                warnmsg = "invalid value encountered"
                for thiswarn in warnlist:

                    if (issubclass(thiswarn.category, RuntimeWarning)
                        and str(thiswarn.message).startswith(warnmsg)):
                        invalid_flag = True

            cfunc(input_operand, result)

            # Need special checks if NaNs are in results
            if np.isnan(expected).any() or np.isnan(result).any():
                self.assertTrue(np.allclose(np.isnan(result), np.isnan(expected)))
                if not np.isnan(expected).all() and not np.isnan(result).all():
                    self.assertTrue(np.allclose(result[np.invert(np.isnan(result))],
                                     expected[np.invert(np.isnan(expected))]))
            else:
                match = np.all(result == expected) or np.allclose(result,
                                                                  expected)
                if not match:
                    if invalid_flag:
                        # Allow output to mismatch for invalid input
                        print("Output mismatch for invalid input",
                              input_tuple, result, expected)
                    else:
                        self.fail("%s != %s" % (result, expected))


    def binary_ufunc_test(self, ufunc_name, flags=enable_pyobj_flags,
                         skip_inputs=[], additional_inputs=[],
                         int_output_type=None, float_output_type=None):

        ufunc = _make_binary_ufunc_usecase(ufunc_name)
#        ufunc = globals()[ufunc_name + '_usecase']

        inputs = list(self.inputs) + additional_inputs
        pyfunc = ufunc

        for input_tuple in inputs:
            input_operand = input_tuple[0]
            input_type = input_tuple[1]

            if input_type in skip_inputs:
                continue

            ty = input_type
            if isinstance(ty, types.Array):
                ty = ty.dtype

            if ty in types.signed_domain:
                if int_output_type:
                    output_type = types.Array(int_output_type, 1, 'C')
                else:
                    output_type = types.Array(types.int64, 1, 'C')
            elif ty in types.unsigned_domain:
                if int_output_type:
                    output_type = types.Array(int_output_type, 1, 'C')
                else:
                    output_type = types.Array(types.uint64, 1, 'C')
            else:
                if float_output_type:
                    output_type = types.Array(float_output_type, 1, 'C')
                else:
                    output_type = types.Array(types.float64, 1, 'C')

            cr = self.cache.compile(pyfunc, (input_type, input_type, output_type),
                                    flags=flags)
            cfunc = cr.entry_point

            if isinstance(input_operand, np.ndarray):
                result = np.zeros(input_operand.size,
                                  dtype=output_type.dtype.name)
                expected = np.zeros(input_operand.size,
                                    dtype=output_type.dtype.name)
            else:
                result = np.zeros(1, dtype=output_type.dtype.name)
                expected = np.zeros(1, dtype=output_type.dtype.name)
            cfunc(input_operand, input_operand, result)
            pyfunc(input_operand, input_operand, expected)

            # Need special checks if NaNs are in results
            if np.isnan(expected).any() or np.isnan(result).any():
                self.assertTrue(np.allclose(np.isnan(result), np.isnan(expected)))
                if not np.isnan(expected).all() and not np.isnan(result).all():
                    self.assertTrue(np.allclose(result[np.invert(np.isnan(result))],
                                     expected[np.invert(np.isnan(expected))]))
            else:
                self.assertTrue(np.all(result == expected) or
                                np.allclose(result, expected))


    def unary_int_ufunc_test(self, name=None, flags=enable_pyobj_flags):
        self.unary_ufunc_test(name, flags=flags,
            skip_inputs=[types.float32, types.float64,
                types.Array(types.float32, 1, 'C'),
                types.Array(types.float64, 1, 'C')])

    def binary_int_ufunc_test(self, name=None, flags=enable_pyobj_flags):
        self.binary_ufunc_test(name, flags=flags,
            skip_inputs=[types.float32, types.float64,
                types.Array(types.float32, 1, 'C'),
                types.Array(types.float64, 1, 'C')])


    ############################################################################
    # Math operations
    def test_add_ufunc(self, flags=enable_pyobj_flags):
        self.binary_ufunc_test('add', flags=flags)

    def test_add_ufunc_npm(self):
        self.test_add_ufunc(flags=no_pyobj_flags)

    def test_subtract_ufunc(self, flags=enable_pyobj_flags):
        self.binary_ufunc_test('subtract', flags=flags)

    def test_subtract_ufunc_npm(self):
        self.test_subtract_ufunc(flags=no_pyobj_flags)

    def test_multiply_ufunc(self, flags=enable_pyobj_flags):
        self.binary_ufunc_test('multiply', flags=flags)

    def test_multiply_ufunc_npm(self):
        self.test_multiply_ufunc(flags=no_pyobj_flags)

    def test_divide_ufunc(self, flags=enable_pyobj_flags):
        # Bear in mind that in python3 divide IS true_divide
        # so the out type for int types will be a double
        int_out_type = None
        if PYVERSION >= (3, 0):
            int_out_type = types.float64

        self.binary_ufunc_test('divide', flags=flags, int_output_type=int_out_type)

    def test_divide_ufunc_npm(self):
        self.test_divide_ufunc(flags=no_pyobj_flags)

    def test_logaddexp_ufunc(self):
        self.binary_ufunc_test('logaddexp')

    def test_logaddexp_ufunc_npm(self):
        self.binary_ufunc_test('logaddexp', flags=no_pyobj_flags)

    def test_logaddexp2_ufunc(self):
        self.binary_ufunc_test('logaddexp2')

    def test_logaddexp2_ufunc_npm(self):
        self.binary_ufunc_test('logaddexp2', flags=no_pyobj_flags)

    def test_true_divide_ufunc(self, flags=enable_pyobj_flags):
        self.binary_ufunc_test('true_divide', flags=flags, int_output_type=types.float64)

    def test_true_divide_ufunc_npm(self):
        self.test_true_divide_ufunc(flags=no_pyobj_flags)

    def test_floor_divide_ufunc(self):
        self.binary_ufunc_test('floor_divide')

    def test_floor_divide_ufunc_npm(self):
        self.binary_ufunc_test('floor_divide', flags=no_pyobj_flags)

    def test_negative_ufunc(self, flags=enable_pyobj_flags):
        # NumPy ufunc has bug with uint32 as input and int64 as output,
        # so skip uint32 input.
        self.unary_ufunc_test('negative', int_output_type=types.int64,
                              skip_inputs=[types.Array(types.uint32, 1, 'C'), types.uint32], flags=flags)

    def test_negative_ufunc_npm(self):
        self.test_negative_ufunc(flags=no_pyobj_flags)

    def test_power_ufunc(self):
        self.binary_ufunc_test('power')

    def test_power_ufunc_npm(self):
        self.binary_ufunc_test('power', flags=no_pyobj_flags)

    def test_remainder_ufunc(self):
        self.binary_ufunc_test('remainder')

    @_unimplemented
    def test_remainder_ufunc_npm(self):
        self.binary_ufunc_test('remainder', flags=no_pyobj_flags)

    def test_mod_ufunc(self):
        self.binary_ufunc_test('mod')

    @_unimplemented
    def test_mod_ufunc_npm(self):
        self.binary_ufunc_test('mod', flags=no_pyobj_flags)

    def test_fmod_ufunc(self):
        self.binary_ufunc_test('fmod')

    @_unimplemented
    def test_fmod_ufunc_npm(self):
        self.binary_ufunc_test('fmod', flags=no_pyobj_flags)

    def test_abs_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('abs', flags=flags,
            additional_inputs = [(np.iinfo(np.uint32).max, types.uint32),
                                 (np.iinfo(np.uint64).max, types.uint64),
                                 (np.finfo(np.float32).min, types.float32),
                                 (np.finfo(np.float64).min, types.float64)
                                 ])

    def test_abs_ufunc_npm(self):
        self.test_abs_ufunc(flags=no_pyobj_flags)

    def test_absolute_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('absolute', flags=flags,
            additional_inputs = [(np.iinfo(np.uint32).max, types.uint32),
                                 (np.iinfo(np.uint64).max, types.uint64),
                                 (np.finfo(np.float32).min, types.float32),
                                 (np.finfo(np.float64).min, types.float64)
                                 ])

    def test_absolute_ufunc_npm(self):
        self.test_absolute_ufunc(flags=no_pyobj_flags)

    def test_fabs_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('fabs', flags=flags)

    def test_fabs_ufunc_npm(self):
        self.test_fabs_ufunc(flags=no_pyobj_flags)

    def test_rint_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('rint', flags=flags)

    def test_rint_ufunc_npm(self):
        self.test_rint_ufunc(flags=no_pyobj_flags)

    def test_sign_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('sign', flags=flags)

    def test_sign_ufunc_npm(self):
        self.test_sign_ufunc(flags=no_pyobj_flags)

    def test_conj_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('conj', flags=flags)

    @_unimplemented
    def test_conj_ufunc_npm(self):
        self.test_conj_ufunc(flags=no_pyobj_flags)

    def test_exp_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('exp', flags=flags)

    def test_exp_ufunc_npm(self):
        self.test_exp_ufunc(flags=no_pyobj_flags)

    def test_exp2_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('exp2', flags=flags)

    def test_exp2_ufunc_npm(self):
        self.test_exp2_ufunc(flags=no_pyobj_flags)

    def test_log_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('log', flags=flags)

    def test_log_ufunc_npm(self):
        self.test_log_ufunc(flags=no_pyobj_flags)

    def test_log2_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('log2', flags=flags)

    def test_log2_ufunc_npm(self):
        self.test_log2_ufunc(flags=no_pyobj_flags)

    def test_log10_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('log10', flags=flags)

    def test_log10_ufunc_npm(self):
        self.test_log10_ufunc(flags=no_pyobj_flags)

    def test_expm1_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('expm1', flags=flags)

    def test_expm1_ufunc_npm(self):
        self.test_expm1_ufunc(flags=no_pyobj_flags)

    def test_log1p_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('log1p', flags=flags)

    def test_log1p_ufunc_npm(self):
        self.test_log1p_ufunc(flags=no_pyobj_flags)

    def test_sqrt_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('sqrt', flags=flags)

    def test_sqrt_ufunc_npm(self):
        self.test_sqrt_ufunc(flags=no_pyobj_flags)

    def test_square_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('square', flags=flags)

    @_unimplemented
    def test_square_ufunc_npm(self):
        self.test_square_ufunc(flags=no_pyobj_flags)

    def test_reciprocal_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('reciprocal', flags=flags)

    @_unimplemented
    def test_reciprocal_ufunc_npm(self):
        self.test_reciprocal_ufunc(flags=no_pyobj_flags)

    def test_conjugate_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('reciprocal', flags=flags)

    @_unimplemented
    def test_conjugate_ufunc_npm(self):
        self.test_reciprocal_ufunc(flags=no_pyobj_flags)


    ############################################################################
    # Trigonometric Functions

    def test_sin_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('sin', flags=flags)

    def test_sin_ufunc_npm(self):
        self.test_sin_ufunc(flags=no_pyobj_flags)

    def test_cos_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('cos', flags=flags)

    def test_cos_ufunc_npm(self):
        self.test_cos_ufunc(flags=no_pyobj_flags)

    def test_tan_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('tan', flags=flags)

    def test_tan_ufunc_npm(self):
        self.test_tan_ufunc(flags=no_pyobj_flags)

    def test_arcsin_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('arcsin', flags=flags)

    def test_arcsin_ufunc_npm(self):
        self.test_arcsin_ufunc(flags=no_pyobj_flags)

    def test_arccos_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('arccos', flags=flags)

    def test_arccos_ufunc_npm(self):
        self.test_arccos_ufunc(flags=no_pyobj_flags)

    def test_arctan_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('arctan', flags=flags)

    def test_arctan_ufunc_npm(self):
        self.test_arctan_ufunc(flags=no_pyobj_flags)

    def test_arctan2_ufunc(self):
        self.binary_ufunc_test('arctan2')

    def test_arctan2_ufunc_npm(self):
        self.binary_ufunc_test('arctan2', flags=no_pyobj_flags)

    def test_hypot_ufunc(self):
        self.binary_ufunc_test('hypot')

    def test_hypot_ufunc_npm(self):
        self.binary_ufunc_test('hypot', flags=no_pyobj_flags)

    def test_sinh_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('sinh', flags=flags)

    def test_sinh_ufunc_npm(self):
        self.test_sinh_ufunc(flags=no_pyobj_flags)

    def test_cosh_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('cosh', flags=flags)

    def test_cosh_ufunc_npm(self):
        self.test_cosh_ufunc(flags=no_pyobj_flags)

    def test_tanh_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('tanh', flags=flags)

    def test_tanh_ufunc_npm(self):
        self.test_tanh_ufunc(flags=no_pyobj_flags)

    def test_arcsinh_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('arcsinh', flags=flags)

    def test_arcsinh_ufunc_npm(self):
        self.test_arcsinh_ufunc(flags=no_pyobj_flags)

    def test_arccosh_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('arccosh', flags=flags)

    def test_arccosh_ufunc_npm(self):
        self.test_arccosh_ufunc(flags=no_pyobj_flags)

    def test_arctanh_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('arctanh', flags=flags)

    def test_arctanh_ufunc_npm(self):
        self.test_arctanh_ufunc(flags=no_pyobj_flags)

    def test_deg2rad_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('deg2rad', flags=flags)

    def test_deg2rad_ufunc_npm(self):
        self.test_deg2rad_ufunc(flags=no_pyobj_flags)

    def test_rad2deg_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('rad2deg', flags=flags)

    def test_rad2deg_ufunc_npm(self):
        self.test_rad2deg_ufunc(flags=no_pyobj_flags)

    def test_degrees_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('degrees', flags=flags)

    def test_degrees_ufunc_npm(self):
        self.test_degrees_ufunc(flags=no_pyobj_flags)

    def test_radians_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('radians', flags=flags)

    def test_radians_ufunc_npm(self):
        self.test_radians_ufunc(flags=no_pyobj_flags)


    ############################################################################
    # Bit-twiddling Functions

    def test_bitwise_and_ufunc(self):
        self.binary_int_ufunc_test('bitwise_and')

    @_unimplemented
    def test_bitwise_and_ufunc_npm(self):
        self.binary_int_ufunc_test('bitwise_and', flags=no_pyobj_flags)

    def test_bitwise_or_ufunc(self):
        self.binary_int_ufunc_test('bitwise_or')

    @_unimplemented
    def test_bitwise_or_ufunc_npm(self):
        self.binary_int_ufunc_test('bitwise_or', flags=no_pyobj_flags)

    def test_bitwise_xor_ufunc(self):
        self.binary_int_ufunc_test('bitwise_xor')

    @_unimplemented
    def test_bitwise_xor_ufunc_npm(self):
        self.binary_int_ufunc_test('bitwise_xor', flags=no_pyobj_flags)

    def test_invert_ufunc(self, flags=enable_pyobj_flags):
        self.unary_int_ufunc_test('invert', flags=flags)

    @_unimplemented
    def test_invert_ufunc_npm(self):
        self.test_invert_ufunc(flags=no_pyobj_flags)

    def test_bitwise_not_ufunc(self, flags=enable_pyobj_flags):
        self.unary_int_ufunc_test('bitwise_not', flags=flags)

    @_unimplemented
    def test_bitwise_not_ufunc_npm(self):
        self.test_bitwise_not_ufunc(flags=no_pyobj_flags)

    def test_left_shift_ufunc(self):
        self.binary_int_ufunc_test('left_shift')

    @_unimplemented
    def test_left_shift_ufunc_npm(self):
        self.binary_int_ufunc_test('left_shift', flags=no_pyobj_flags)

    def test_right_shift_ufunc(self):
        self.binary_int_ufunc_test('right_shift')

    @_unimplemented
    def test_right_shift_ufunc_npm(self):
        self.binary_int_ufunc_test('right_shift', flags=no_pyobj_flags)


    ############################################################################
    # Comparison functions
    def test_greater_ufunc(self):
        self.binary_ufunc_test('greater')

    @_unimplemented
    def test_greater_ufunc_npm(self):
        self.binary_ufunc_test('greater', flags=no_pyobj_flags)

    def test_greater_equal_ufunc(self):
        self.binary_ufunc_test('greater_equal')

    @_unimplemented
    def test_greater_equal_ufunc_npm(self):
        self.binary_ufunc_test('greater_equal', flags=no_pyobj_flags)

    def test_less_ufunc(self):
        self.binary_ufunc_test('less')

    @_unimplemented
    def test_less_ufunc_npm(self):
        self.binary_ufunc_test('less', flags=no_pyobj_flags)

    def test_less_equal_ufunc(self):
        self.binary_ufunc_test('less_equal')

    @_unimplemented
    def test_less_equal_ufunc_npm(self):
        self.binary_ufunc_test('less_equal', flags=no_pyobj_flags)

    def test_not_equal_ufunc(self):
        self.binary_ufunc_test('not_equal')

    @_unimplemented
    def test_not_equal_ufunc_npm(self):
        self.binary_ufunc_test('not_equal', flags=no_pyobj_flags)

    def test_equal_ufunc(self):
        self.binary_ufunc_test('equal')

    @_unimplemented
    def test_equal_ufunc_npm(self):
        self.binary_ufunc_test('equal', flags=no_pyobj_flags)

    def test_logical_and_ufunc(self):
        self.binary_ufunc_test('logical_and')

    @_unimplemented
    def test_logical_and_ufunc_npm(self):
        self.binary_ufunc_test('logical_and', flags=no_pyobj_flags)

    def test_logical_or_ufunc(self):
        self.binary_ufunc_test('logical_or')

    @_unimplemented
    def test_logical_or_ufunc_npm(self):
        self.binary_ufunc_test('logical_or', flags=no_pyobj_flags)

    def test_logical_xor_ufunc(self):
        self.binary_ufunc_test('logical_xor')

    @_unimplemented
    def test_logical_xor_ufunc_npm(self):
        self.binary_ufunc_test('logical_xor', flags=no_pyobj_flags)

    #logical_not support should be coming soon
    def test_logical_not_ufunc(self):
        self.unary_ufunc_test('logical_not')

    @_unimplemented
    def test_logical_not_ufunc_npm(self):
        self.unary_ufunc_test('logical_not', flags=no_pyobj_flags)

    def test_maximum_ufunc(self):
        self.binary_ufunc_test('maximum')

    @_unimplemented
    def test_maximum_ufunc_npm(self):
        self.binary_ufunc_test('maximum', flags=no_pyobj_flags)

    def test_minimum_ufunc(self):
        self.binary_ufunc_test('minimum')

    @_unimplemented
    def test_minimum_ufunc_npm(self):
        self.binary_ufunc_test('minimum', flags=no_pyobj_flags)

    def test_fmax_ufunc(self):
        self.binary_ufunc_test('fmax')

    @_unimplemented
    def test_fmax_ufunc_npm(self):
        self.binary_ufunc_test('fmax', flags=no_pyobj_flags)

    def test_fmin_ufunc(self):
        self.binary_ufunc_test('fmin')

    @_unimplemented
    def test_fmin_ufunc_npm(self):
        self.binary_ufunc_test('fmin', flags=no_pyobj_flags)


    ############################################################################
    # Floating functions
    def test_isfinite_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('isfinite', flags=flags)

    @_unimplemented
    def test_isfinite_ufunc_npm(self):
        self.test_isfinite_ufunc(flags=no_pyobj_flags)

    def test_isinf_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('isinf', flags=flags)

    @_unimplemented
    def test_isinf_ufunc_npm(self):
        self.test_isinf_ufunc(flags=no_pyobj_flags)

    def test_isnan_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('isnan', flags=flags)

    @_unimplemented
    def test_isnan_ufunc_npm(self):
        self.test_isnan_ufunc(flags=no_pyobj_flags)

    def test_signbit_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('signbit', flags=flags)

    @_unimplemented
    def test_signbit_ufunc_npm(self):
        self.test_signbit_ufunc(flags=no_pyobj_flags)

    def test_copysign_ufunc(self, flags=enable_pyobj_flags):
        self.binary_ufunc_test('copysign')

    def test_copysign_ufunc_npm(self):
        self.test_copysign_ufunc(flags=no_pyobj_flags)

    @_unimplemented
    def test_nextafter_ufunc(self, flags=enable_pyobj_flags):
        self.binary_ufunc_test('nextafter', flags=flags)

    @_unimplemented
    def test_nextafter_ufunc_npm(self):
        self.test_nextafter_ufunc(flags=no_pyobj_flags)

    def test_modf_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('modf', flags=flags)

    @_unimplemented
    def test_modf_ufunc_npm(self):
        self.test_modf_ufunc(flags=no_pyobj_flags)


    # FIXME - ldexp does not have homogeneous arguments, so the usual tests won't
    #         work as they reuse both inputs
    @unittest.skipIf(is32bits or iswindows, "Some types are not supported on "
                                       "32-bit "
                               "platform")
    @unittest.skip("this test needs fixing")
    def test_ldexp_ufunc(self):
        self.binary_int_ufunc_test('ldexp')

    # FIXME
    @unittest.skipIf(is32bits or iswindows,
                     "Some types are not supported on 32-bit platform")
    @unittest.skip("this tests needs fixing")
    def test_ldexp_ufunc_npm(self):
        self.binary_int_ufunc_test('ldexp', flags=no_pyobj_flags)

    def test_frexp_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('frexp', flags=flags)

    @_unimplemented
    def test_frexp_ufunc_npm(self):
        self.test_frexp_ufunc(flags=no_pyobj_flags)

    def test_floor_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('floor', flags=flags)

    def test_floor_ufunc_npm(self):
        self.test_floor_ufunc(flags=no_pyobj_flags)

    def test_ceil_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('ceil', flags=flags)

    def test_ceil_ufunc_npm(self):
        self.test_ceil_ufunc(flags=no_pyobj_flags)

    def test_trunc_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('trunc', flags=flags)

    def test_trunc_ufunc_npm(self):
        self.test_trunc_ufunc(flags=no_pyobj_flags)

    def test_spacing_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test('spacing', flags=flags)

    @_unimplemented
    def test_spacing_ufunc_npm(self):
        self.test_spacing_ufunc(flags=nopyobj_flags)

    ############################################################################
    # Other tests
    def test_binary_ufunc_performance(self):

        pyfunc = _make_binary_ufunc_usecase('add')
        arraytype = types.Array(types.float32, 1, 'C')
        cr = compile_isolated(pyfunc, (arraytype, arraytype, arraytype))
        cfunc = cr.entry_point

        nelem = 5000
        x_operand = np.arange(nelem, dtype=np.float32)
        y_operand = np.arange(nelem, dtype=np.float32)
        control = np.empty_like(x_operand)
        result = np.empty_like(x_operand)

        def bm_python():
            pyfunc(x_operand, y_operand, control)

        def bm_numba():
            cfunc(x_operand, y_operand, result)

        print(utils.benchmark(bm_python, maxsec=.1))
        print(utils.benchmark(bm_numba, maxsec=.1))
        assert np.allclose(control, result)

    def binary_ufunc_mixed_types_test(self, ufunc_name, flags=enable_pyobj_flags):
        ufunc = _make_binary_ufunc_usecase(ufunc_name)
        #ufunc = globals()[ufunc_name + '_usecase']

        inputs1 = [
            (1, types.uint64),
            (-1, types.int64),
            (0.5, types.float64),

            (np.array([0, 1], dtype='u8'), types.Array(types.uint64, 1, 'C')),
            (np.array([-1, 1], dtype='i8'), types.Array(types.int64, 1, 'C')),
            (np.array([-0.5, 0.5], dtype='f8'), types.Array(types.float64, 1, 'C'))]

        inputs2 = inputs1

        output_types = [types.Array(types.int64, 1, 'C'),
                        types.Array(types.float64, 1, 'C')]

        pyfunc = ufunc

        for input1, input2, output_type in itertools.product(inputs1, inputs2, output_types):

            input1_operand = input1[0]
            input1_type = input1[1]

            input2_operand = input2[0]
            input2_type = input2[1]

            # Skip division by unsigned int because of NumPy bugs
            if ufunc_name == 'divide' and (input2_type == types.Array(types.uint32, 1, 'C') or
                    input2_type == types.Array(types.uint64, 1, 'C')):
                continue

            # Skip some subtraction tests because of NumPy bugs
            if ufunc_name == 'subtract' and input1_type == types.Array(types.uint32, 1, 'C') and \
                    input2_type == types.uint32 and types.Array(types.int64, 1, 'C'):
                continue
            if ufunc_name == 'subtract' and input1_type == types.Array(types.uint32, 1, 'C') and \
                    input2_type == types.uint64 and types.Array(types.int64, 1, 'C'):
                continue

            if ((isinstance(input1_type, types.Array) or
                    isinstance(input2_type, types.Array)) and
                    not isinstance(output_type, types.Array)):
                continue

            cr = self.cache.compile(pyfunc,
                                    (input1_type, input2_type, output_type),
                                    flags=flags)
            cfunc = cr.entry_point

            if isinstance(input1_operand, np.ndarray):
                result = np.zeros(input1_operand.size,
                                  dtype=output_type.dtype.name)
                expected = np.zeros(input1_operand.size,
                                    dtype=output_type.dtype.name)
            elif isinstance(input2_operand, np.ndarray):
                result = np.zeros(input2_operand.size,
                                  dtype=output_type.dtype.name)
                expected = np.zeros(input2_operand.size,
                                    dtype=output_type.dtype.name)
            else:
                result = np.zeros(1, dtype=output_type.dtype.name)
                expected = np.zeros(1, dtype=output_type.dtype.name)

            cfunc(input1_operand, input2_operand, result)
            pyfunc(input1_operand, input2_operand, expected)

            # Need special checks if NaNs are in results
            if np.isnan(expected).any() or np.isnan(result).any():
                self.assertTrue(np.allclose(np.isnan(result), np.isnan(expected)))
                if not np.isnan(expected).all() and not np.isnan(result).all():
                    self.assertTrue(np.allclose(result[np.invert(np.isnan(result))],
                                     expected[np.invert(np.isnan(expected))]))
            else:
                self.assertTrue(np.all(result == expected) or
                                np.allclose(result, expected))

    def test_mixed_types(self):
        self.binary_ufunc_mixed_types_test('divide', flags=no_pyobj_flags)


    def test_broadcasting(self):

        # Test unary ufunc
        pyfunc = _make_unary_ufunc_usecase('negative')

        input_operands = [
            np.arange(3, dtype='i8'),
            np.arange(3, dtype='i8').reshape(3,1),
            np.arange(3, dtype='i8').reshape(1,3),
            np.arange(3, dtype='i8').reshape(3,1),
            np.arange(3, dtype='i8').reshape(1,3),
            np.arange(3*3, dtype='i8').reshape(3,3)]

        output_operands = [
            np.zeros(3*3, dtype='i8').reshape(3,3),
            np.zeros(3*3, dtype='i8').reshape(3,3),
            np.zeros(3*3, dtype='i8').reshape(3,3),
            np.zeros(3*3*3, dtype='i8').reshape(3,3,3),
            np.zeros(3*3*3, dtype='i8').reshape(3,3,3),
            np.zeros(3*3*3, dtype='i8').reshape(3,3,3)]

        for x, result in zip(input_operands, output_operands):

            input_type = types.Array(types.uint64, x.ndim, 'C')
            output_type = types.Array(types.int64, result.ndim, 'C')

            cr = self.cache.compile(pyfunc, (input_type, output_type),
                                    flags=no_pyobj_flags)
            cfunc = cr.entry_point

            expected = np.zeros(result.shape, dtype=result.dtype)
            np.negative(x, expected)

            cfunc(x, result)
            self.assertTrue(np.all(result == expected))

        # Test binary ufunc
        pyfunc = _make_binary_ufunc_usecase('add')

        input1_operands = [
            np.arange(3, dtype='u8'),
            np.arange(3*3, dtype='u8').reshape(3,3),
            np.arange(3*3*3, dtype='u8').reshape(3,3,3),
            np.arange(3, dtype='u8').reshape(3,1),
            np.arange(3, dtype='u8').reshape(1,3),
            np.arange(3, dtype='u8').reshape(3,1,1),
            np.arange(3*3, dtype='u8').reshape(3,3,1),
            np.arange(3*3, dtype='u8').reshape(3,1,3),
            np.arange(3*3, dtype='u8').reshape(1,3,3)]

        input2_operands = input1_operands

        for x, y in itertools.product(input1_operands, input2_operands):

            input1_type = types.Array(types.uint64, x.ndim, 'C')
            input2_type = types.Array(types.uint64, y.ndim, 'C')
            output_type = types.Array(types.uint64, max(x.ndim, y.ndim), 'C')

            cr = self.cache.compile(pyfunc, (input1_type, input2_type, output_type),
                                    flags=no_pyobj_flags)
            cfunc = cr.entry_point

            expected = np.add(x, y)
            result = np.zeros(expected.shape, dtype='u8')

            cfunc(x, y, result)
            self.assertTrue(np.all(result == expected))


class TestScalarUFuncs(TestCase):
    """check the machinery of ufuncs works when the result is an scalar.
    These are not exhaustive because:
    - the machinery to support this case is the same for all the functions of a
      given arity.
    - the result of the inner function itself is already tested in TestUFuncs
    """
    def run_ufunc(self, pyfunc, arg_types, arg_values, flags=enable_pyobj_flags):
        for tyargs, args in zip(arg_types, arg_values):
            cr = compile_isolated(pyfunc, tyargs, flags=flags)
            cfunc = cr.entry_point
            got = cfunc(*args)
            expected = pyfunc(*_as_dtype_value(tyargs, args))

            msg = 'for args {0} typed {1}'.format(args, tyargs)

            # note: due to semantics of ufuncs, thing like adding a int32 to a
            # uint64 results in doubles (as neither int32 can be cast safely
            # to uint64 nor vice-versa, falling back to using the float version.
            # Modify in those cases the expected value (the numpy version does
            # not use typed integers as inputs so its result is an integer)
            special = set([(types.int32, types.uint64), (types.uint64, types.int32),
                           (types.int64, types.uint64), (types.uint64, types.int64)])
            if tyargs in special:
                expected = float(expected)
            else:
                # The numba version of scalar ufuncs return an actual value that
                # gets converted to a Python type, instead of using NumPy scalars.
                # although in python 2 NumPy scalars are considered and instance of
                # the appropriate python type, in python 3 that is no longer the case.
                # This is why the expected result is casted to the appropriate Python
                # type (which is actually the expected behavior of the ufunc translation)
                if np.issubdtype(expected.dtype, np.inexact):
                    expected = float(expected)
                elif np.issubdtype(expected.dtype, np.integer):
                    expected = int(expected)
                elif np.issubdtype(expected.dtype, np.bool):
                    expected = bool(expected)

            alltypes = cr.signature.args + (cr.signature.return_type,)

            # select the appropriate precision for comparison: note that an argument
            # typed at a lower precision can introduce precision problems. For this
            # reason the argument types must be taken into account.
            if any([t==types.float32 for t in alltypes]):
                prec='single'
            elif any([t==types.float64 for t in alltypes]):
                prec='double'
            else:
                prec='exact'

            self.assertPreciseEqual(got, expected, msg=msg, prec=prec)


    def test_scalar_unary_ufunc(self, flags=enable_pyobj_flags):
        def _func(x):
            return np.sqrt(x)

        vals = [(2,), (2,), (1,), (2,), (.1,), (.2,)]
        tys = [(types.int32,), (types.uint32,),
               (types.int64,), (types.uint64,), (types.float32,), (types.float64,)]
        self.run_ufunc(_func, tys, vals, flags=flags)

    def test_scalar_unary_ufunc_npm(self):
        self.test_scalar_unary_ufunc(flags=no_pyobj_flags)


    def test_scalar_binary_uniform_ufunc(self, flags=enable_pyobj_flags):
        def _func(x,y):
            return np.add(x,y)

        vals = [2, 2, 1, 2, .1, .2]
        tys = [types.int32, types.uint32,
               types.int64, types.uint64, types.float32, types.float64]
        self.run_ufunc(_func, zip(tys, tys), zip(vals, vals), flags=flags)

    def test_scalar_binary_uniform_ufuncs_npm(self):
        self.test_scalar_binary_uniform_ufunc(flags=no_pyobj_flags)

    def test_scalar_binary_mixed_ufunc(self, flags=enable_pyobj_flags):
        def _func(x,y):
            return np.add(x,y)

        vals = [2, 2, 1, 2, .1, .2]
        tys = [types.int32, types.uint32,
               types.int64, types.uint64, types.float32, types.float64]
        self.run_ufunc(_func, itertools.product(tys, tys), itertools.product(vals, vals),
                       flags=flags)

    def test_scalar_binary_mixed_ufuncs_npm(self):
        self.test_scalar_binary_mixed_ufunc(flags=no_pyobj_flags)


class TestUfuncIssues(TestCase):
    def test_issue_651(self):
        # Exercise the code path to make sure this does not fail
        @vectorize(["(float64,float64)"])
        def foo(x1, x2):
            return np.add(x1, x2) + np.add(x1, x2)

        a = np.arange(10, dtype='f8')
        b = np.arange(10, dtype='f8')
        self.assertTrue(np.all(foo(a, b) == (a + b) + (a + b)))


if __name__ == '__main__':
    unittest.main()
