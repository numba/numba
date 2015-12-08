from __future__ import print_function

import functools
import itertools
import re
import sys
import warnings

import numpy as np

import numba.unittest_support as unittest
from numba import types, typing, utils, typeof, numpy_support, njit
from numba.compiler import compile_isolated, Flags, DEFAULT_FLAGS
from numba.numpy_support import from_dtype
from numba import vectorize
from numba.config import PYVERSION
from numba.errors import LoweringError, TypingError
from .support import (TestCase, CompilationCache, skip_on_numpy_16,
                      is_on_numpy_16, MemoryLeakMixin)

from numba.typing.npydecl import supported_ufuncs, all_ufuncs

is32bits = tuple.__itemsize__ == 4
iswindows = sys.platform.startswith('win32')

enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

no_pyobj_flags = Flags()

enable_nrt_flags = Flags()
enable_nrt_flags.set("nrt")


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

def _make_ufunc_usecase(ufunc):
    ldict = {}
    arg_str = ','.join(['a{0}'.format(i) for i in range(ufunc.nargs)])
    func_str = 'def fn({0}):\n    np.{1}({0})'.format(arg_str, ufunc.__name__)
    exec(func_str, globals(), ldict)
    fn = ldict['fn']
    fn.__name__ = '{0}_usecase'.format(ufunc.__name__)
    return fn


def _make_unary_ufunc_usecase(ufunc):
    ufunc_name = ufunc.__name__
    ldict = {}
    exec("def fn(x,out):\n    np.{0}(x,out)".format(ufunc_name), globals(), ldict)
    fn = ldict["fn"]
    fn.__name__ = "{0}_usecase".format(ufunc_name)
    return fn


def _make_unary_ufunc_op_usecase(ufunc_op):
    ldict = {}
    exec("def fn(x):\n    return {0}(x)".format(ufunc_op), globals(), ldict)
    fn = ldict["fn"]
    fn.__name__ = "usecase_{0}".format(hash(ufunc_op))
    return fn


def _make_binary_ufunc_usecase(ufunc):
    ufunc_name = ufunc.__name__
    ldict = {}
    exec("def fn(x,y,out):\n    np.{0}(x,y,out)".format(ufunc_name), globals(), ldict);
    fn = ldict['fn']
    fn.__name__ = "{0}_usecase".format(ufunc_name)
    return fn


def _make_binary_ufunc_op_usecase(ufunc_op):
    ldict = {}
    exec("def fn(x,y):\n    return x{0}y".format(ufunc_op), globals(), ldict)
    fn = ldict["fn"]
    fn.__name__ = "usecase_{0}".format(hash(ufunc_op))
    return fn


def _make_inplace_ufunc_op_usecase(ufunc_op):
    ldict = {}
    exec("def fn(x,y):\n    x{0}y".format(ufunc_op), globals(), ldict)
    fn = ldict["fn"]
    fn.__name__ = "usecase_{0}".format(hash(ufunc_op))
    return fn


def _as_dtype_value(tyargs, args):
    """Convert python values into numpy scalar objects.
    """
    return [np.dtype(str(ty)).type(val) for ty, val in zip(tyargs, args)]



class BaseUFuncTest(MemoryLeakMixin):

    def setUp(self):
        super(BaseUFuncTest, self).setUp()
        self.inputs = [
            (np.uint32(0), types.uint32),
            (np.uint32(1), types.uint32),
            (np.int32(-1), types.int32),
            (np.int32(0), types.int32),
            (np.int32(1), types.int32),
            (np.uint64(0), types.uint64),
            (np.uint64(1), types.uint64),
            (np.int64(-1), types.int64),
            (np.int64(0), types.int64),
            (np.int64(1), types.int64),

            (np.float32(-0.5), types.float32),
            (np.float32(0.0), types.float32),
            (np.float32(0.5), types.float32),

            (np.float64(-0.5), types.float64),
            (np.float64(0.0), types.float64),
            (np.float64(0.5), types.float64),

            (np.array([0,1], dtype='u4'), types.Array(types.uint32, 1, 'C')),
            (np.array([0,1], dtype='u8'), types.Array(types.uint64, 1, 'C')),
            (np.array([-1,0,1], dtype='i4'), types.Array(types.int32, 1, 'C')),
            (np.array([-1,0,1], dtype='i8'), types.Array(types.int64, 1, 'C')),
            (np.array([-0.5, 0.0, 0.5], dtype='f4'), types.Array(types.float32, 1, 'C')),
            (np.array([-0.5, 0.0, 0.5], dtype='f8'), types.Array(types.float64, 1, 'C')),
            ]
        self.cache = CompilationCache()

    def _determine_output_type(self, input_type, int_output_type=None,
                               float_output_type=None):
        ty = input_type
        if isinstance(ty, types.Array):
            ty = ty.dtype

        if ty in types.signed_domain:
            if int_output_type:
                output_type = types.Array(int_output_type, 1, 'C')
            else:
                output_type = types.Array(ty, 1, 'C')
        elif ty in types.unsigned_domain:
            if int_output_type:
                output_type = types.Array(int_output_type, 1, 'C')
            else:
                output_type = types.Array(ty, 1, 'C')
        else:
            if float_output_type:
                output_type = types.Array(float_output_type, 1, 'C')
            else:
                output_type = types.Array(ty, 1, 'C')
        return output_type


class TestUFuncs(BaseUFuncTest, TestCase):

    def unary_ufunc_test(self, ufunc, flags=enable_pyobj_flags,
                         skip_inputs=[], additional_inputs=[],
                         int_output_type=None, float_output_type=None,
                         kinds='ifc'):
        # Necessary to avoid some Numpy warnings being silenced, despite
        # the simplefilter() call below.
        self.reset_module_warnings(__name__)

        ufunc = _make_unary_ufunc_usecase(ufunc)

        inputs = list(self.inputs)
        inputs.extend(additional_inputs)

        pyfunc = ufunc

        for input_tuple in inputs:
            input_operand = input_tuple[0]
            input_type = input_tuple[1]

            if input_type in skip_inputs:
                continue
            # Some ufuncs don't allow all kinds of arguments, and implicit
            # conversion has become stricter in 1.10.
            if (numpy_support.strict_ufunc_typing and
                input_operand.dtype.kind not in kinds):
                continue

            output_type = self._determine_output_type(
                input_type, int_output_type, float_output_type)

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
                        msg = '\n'.join(["ufunc '{0}' failed",
                                         "inputs ({1}):", "{2}",
                                         "got({3})", "{4}",
                                         "expected ({5}):", "{6}"
                                     ]).format(ufunc.__name__,
                                               input_type, input_operand,
                                               output_type, result,
                                               expected.dtype, expected)
                        self.fail(msg)

    def binary_ufunc_test(self, ufunc, flags=enable_pyobj_flags,
                         skip_inputs=[], additional_inputs=[],
                         int_output_type=None, float_output_type=None,
                         kinds='ifc'):

        ufunc = _make_binary_ufunc_usecase(ufunc)

        inputs = list(self.inputs) + additional_inputs
        pyfunc = ufunc

        for input_tuple in inputs:
            input_operand = input_tuple[0]
            input_type = input_tuple[1]

            if input_type in skip_inputs:
                continue

            # Some ufuncs don't allow all kinds of arguments, and implicit
            # conversion has become stricter in 1.10.
            if (numpy_support.strict_ufunc_typing and
                input_operand.dtype.kind not in kinds):
                continue

            output_type = self._determine_output_type(
                input_type, int_output_type, float_output_type)

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
            np.testing.assert_array_almost_equal(expected, result)

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
        self.binary_ufunc_test(np.add, flags=flags)

    def test_add_ufunc_npm(self):
        self.test_add_ufunc(flags=no_pyobj_flags)

    def test_subtract_ufunc(self, flags=enable_pyobj_flags):
        self.binary_ufunc_test(np.subtract, flags=flags)

    def test_subtract_ufunc_npm(self):
        self.test_subtract_ufunc(flags=no_pyobj_flags)

    def test_multiply_ufunc(self, flags=enable_pyobj_flags):
        self.binary_ufunc_test(np.multiply, flags=flags)

    def test_multiply_ufunc_npm(self):
        self.test_multiply_ufunc(flags=no_pyobj_flags)

    def test_divide_ufunc(self, flags=enable_pyobj_flags):
        # Bear in mind that in python3 divide IS true_divide
        # so the out type for int types will be a double
        int_out_type = None
        if PYVERSION >= (3, 0):
            int_out_type = types.float64

        self.binary_ufunc_test(np.divide, flags=flags, int_output_type=int_out_type)

    def test_divide_ufunc_npm(self):
        self.test_divide_ufunc(flags=no_pyobj_flags)

    def test_logaddexp_ufunc(self):
        self.binary_ufunc_test(np.logaddexp, kinds='f')

    def test_logaddexp_ufunc_npm(self):
        self.binary_ufunc_test(np.logaddexp, flags=no_pyobj_flags, kinds='f')

    def test_logaddexp2_ufunc(self):
        self.binary_ufunc_test(np.logaddexp2, kinds='f')

    def test_logaddexp2_ufunc_npm(self):
        self.binary_ufunc_test(np.logaddexp2, flags=no_pyobj_flags, kinds='f')

    def test_true_divide_ufunc(self, flags=enable_pyobj_flags):
        self.binary_ufunc_test(np.true_divide, flags=flags, int_output_type=types.float64)

    def test_true_divide_ufunc_npm(self):
        self.test_true_divide_ufunc(flags=no_pyobj_flags)

    def test_floor_divide_ufunc(self):
        self.binary_ufunc_test(np.floor_divide)

    def test_floor_divide_ufunc_npm(self):
        self.binary_ufunc_test(np.floor_divide, flags=no_pyobj_flags)

    def test_negative_ufunc(self, flags=enable_pyobj_flags):
        # NumPy ufunc has bug with uint32 as input and int64 as output,
        # so skip uint32 input.
        self.unary_ufunc_test(np.negative, int_output_type=types.int64,
                              skip_inputs=[types.Array(types.uint32, 1, 'C'), types.uint32],
                              flags=flags)

    def test_negative_ufunc_npm(self):
        self.test_negative_ufunc(flags=no_pyobj_flags)

    def test_power_ufunc(self, flags=enable_pyobj_flags):
        self.binary_ufunc_test(np.power, flags=flags)

    def test_power_ufunc_npm(self):
        self.test_power_ufunc(flags=no_pyobj_flags)

    def test_remainder_ufunc(self, flags=enable_pyobj_flags):
        self.binary_ufunc_test(np.remainder, flags=flags)

    def test_remainder_ufunc_npm(self):
        self.test_remainder_ufunc(flags=no_pyobj_flags)

    def test_mod_ufunc(self, flags=enable_pyobj_flags):
        self.binary_ufunc_test(np.mod, flags=flags)

    def test_mod_ufunc_npm(self):
        self.test_mod_ufunc(flags=no_pyobj_flags)

    def test_fmod_ufunc(self, flags=enable_pyobj_flags):
        self.binary_ufunc_test(np.fmod, flags=flags)

    def test_fmod_ufunc_npm(self):
        self.test_fmod_ufunc(flags=no_pyobj_flags)

    def test_abs_ufunc(self, flags=enable_pyobj_flags, ufunc=np.abs):
        self.unary_ufunc_test(ufunc, flags=flags,
            additional_inputs = [
                (np.uint32(np.iinfo(np.uint32).max), types.uint32),
                (np.uint64(np.iinfo(np.uint64).max), types.uint64),
                (np.float32(np.finfo(np.float32).min), types.float32),
                (np.float64(np.finfo(np.float64).min), types.float64)
                ])

    def test_abs_ufunc_npm(self):
        self.test_abs_ufunc(flags=no_pyobj_flags)

    def test_absolute_ufunc(self, flags=enable_pyobj_flags):
        self.test_abs_ufunc(flags=flags, ufunc=np.absolute)

    def test_absolute_ufunc_npm(self):
        self.test_absolute_ufunc(flags=no_pyobj_flags)

    def test_fabs_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.fabs, flags=flags, kinds='f')

    def test_fabs_ufunc_npm(self):
        self.test_fabs_ufunc(flags=no_pyobj_flags)

    def test_rint_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.rint, flags=flags, kinds='cf')

    def test_rint_ufunc_npm(self):
        self.test_rint_ufunc(flags=no_pyobj_flags)

    def test_sign_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.sign, flags=flags)

    def test_sign_ufunc_npm(self):
        self.test_sign_ufunc(flags=no_pyobj_flags)

    def test_conj_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.conj, flags=flags)

    def test_conj_ufunc_npm(self):
        self.test_conj_ufunc(flags=no_pyobj_flags)

    def test_exp_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.exp, flags=flags, kinds='cf')

    def test_exp_ufunc_npm(self):
        self.test_exp_ufunc(flags=no_pyobj_flags)

    def test_exp2_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.exp2, flags=flags, kinds='cf')

    def test_exp2_ufunc_npm(self):
        self.test_exp2_ufunc(flags=no_pyobj_flags)

    def test_log_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.log, flags=flags, kinds='cf')

    def test_log_ufunc_npm(self):
        self.test_log_ufunc(flags=no_pyobj_flags)

    def test_log2_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.log2, flags=flags, kinds='cf')

    def test_log2_ufunc_npm(self):
        self.test_log2_ufunc(flags=no_pyobj_flags)

    def test_log10_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.log10, flags=flags, kinds='cf')

    def test_log10_ufunc_npm(self):
        self.test_log10_ufunc(flags=no_pyobj_flags)

    def test_expm1_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.expm1, flags=flags, kinds='cf')

    def test_expm1_ufunc_npm(self):
        self.test_expm1_ufunc(flags=no_pyobj_flags)

    def test_log1p_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.log1p, flags=flags, kinds='cf')

    def test_log1p_ufunc_npm(self):
        self.test_log1p_ufunc(flags=no_pyobj_flags)

    def test_sqrt_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.sqrt, flags=flags, kinds='cf')

    def test_sqrt_ufunc_npm(self):
        self.test_sqrt_ufunc(flags=no_pyobj_flags)

    def test_square_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.square, flags=flags)

    def test_square_ufunc_npm(self):
        self.test_square_ufunc(flags=no_pyobj_flags)

    def test_reciprocal_ufunc(self, flags=enable_pyobj_flags):
        # reciprocal for integers doesn't make much sense and is problematic
        # in the case of division by zero, as an inf will overflow float to
        # int conversions, which is undefined behavior.
        to_skip = [types.Array(types.uint32, 1, 'C'), types.uint32,
                   types.Array(types.int32, 1, 'C'), types.int32,
                   types.Array(types.uint64, 1, 'C'), types.uint64,
                   types.Array(types.int64, 1, 'C'), types.int64]
        self.unary_ufunc_test(np.reciprocal, skip_inputs=to_skip, flags=flags)

    def test_reciprocal_ufunc_npm(self):
        self.test_reciprocal_ufunc(flags=no_pyobj_flags)

    def test_conjugate_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.conjugate, flags=flags)

    def test_conjugate_ufunc_npm(self):
        self.test_conjugate_ufunc(flags=no_pyobj_flags)


    ############################################################################
    # Trigonometric Functions

    def test_sin_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.sin, flags=flags, kinds='cf')

    def test_sin_ufunc_npm(self):
        self.test_sin_ufunc(flags=no_pyobj_flags)

    def test_cos_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.cos, flags=flags, kinds='cf')

    def test_cos_ufunc_npm(self):
        self.test_cos_ufunc(flags=no_pyobj_flags)

    def test_tan_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.tan, flags=flags, kinds='cf')

    def test_tan_ufunc_npm(self):
        self.test_tan_ufunc(flags=no_pyobj_flags)

    def test_arcsin_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.arcsin, flags=flags, kinds='cf')

    def test_arcsin_ufunc_npm(self):
        self.test_arcsin_ufunc(flags=no_pyobj_flags)

    def test_arccos_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.arccos, flags=flags, kinds='cf')

    def test_arccos_ufunc_npm(self):
        self.test_arccos_ufunc(flags=no_pyobj_flags)

    def test_arctan_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.arctan, flags=flags, kinds='cf')

    def test_arctan_ufunc_npm(self):
        self.test_arctan_ufunc(flags=no_pyobj_flags)

    def test_arctan2_ufunc(self, flags=enable_pyobj_flags):
        self.binary_ufunc_test(np.arctan2, flags=flags, kinds='cf')

    def test_arctan2_ufunc_npm(self):
        self.test_arctan2_ufunc(flags=no_pyobj_flags)

    def test_hypot_ufunc(self, flags=enable_pyobj_flags):
        self.binary_ufunc_test(np.hypot, kinds='f')

    def test_hypot_ufunc_npm(self):
        self.test_hypot_ufunc(flags=no_pyobj_flags)

    def test_sinh_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.sinh, flags=flags, kinds='cf')

    def test_sinh_ufunc_npm(self):
        self.test_sinh_ufunc(flags=no_pyobj_flags)

    def test_cosh_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.cosh, flags=flags, kinds='cf')

    def test_cosh_ufunc_npm(self):
        self.test_cosh_ufunc(flags=no_pyobj_flags)

    def test_tanh_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.tanh, flags=flags, kinds='cf')

    def test_tanh_ufunc_npm(self):
        self.test_tanh_ufunc(flags=no_pyobj_flags)

    def test_arcsinh_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.arcsinh, flags=flags, kinds='cf')

    def test_arcsinh_ufunc_npm(self):
        self.test_arcsinh_ufunc(flags=no_pyobj_flags)

    def test_arccosh_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.arccosh, flags=flags, kinds='cf')

    def test_arccosh_ufunc_npm(self):
        self.test_arccosh_ufunc(flags=no_pyobj_flags)

    def test_arctanh_ufunc(self, flags=enable_pyobj_flags):
        # arctanh is only valid is only finite in the range ]-1, 1[
        # This means that for any of the integer types it will produce
        # conversion from infinity/-infinity to integer. That's undefined
        # behavior in C, so the results may vary from implementation to
        # implementation. This means that the result from the compiler
        # used to compile NumPy may differ from the result generated by
        # llvm. Skipping the integer types in this test avoids failed
        # tests because of this.
        to_skip = [types.Array(types.uint32, 1, 'C'), types.uint32,
                   types.Array(types.int32, 1, 'C'), types.int32,
                   types.Array(types.uint64, 1, 'C'), types.uint64,
                   types.Array(types.int64, 1, 'C'), types.int64]

        self.unary_ufunc_test(np.arctanh, skip_inputs=to_skip, flags=flags,
                              kinds='cf')

    def test_arctanh_ufunc_npm(self):
        self.test_arctanh_ufunc(flags=no_pyobj_flags)

    def test_deg2rad_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.deg2rad, flags=flags, kinds='f')

    def test_deg2rad_ufunc_npm(self):
        self.test_deg2rad_ufunc(flags=no_pyobj_flags)

    def test_rad2deg_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.rad2deg, flags=flags, kinds='f')

    def test_rad2deg_ufunc_npm(self):
        self.test_rad2deg_ufunc(flags=no_pyobj_flags)

    def test_degrees_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.degrees, flags=flags, kinds='f')

    def test_degrees_ufunc_npm(self):
        self.test_degrees_ufunc(flags=no_pyobj_flags)

    def test_radians_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.radians, flags=flags, kinds='f')

    def test_radians_ufunc_npm(self):
        self.test_radians_ufunc(flags=no_pyobj_flags)

    ############################################################################
    # Bit-twiddling Functions

    def test_bitwise_and_ufunc(self, flags=enable_pyobj_flags):
        self.binary_int_ufunc_test(np.bitwise_and, flags=flags)

    def test_bitwise_and_ufunc_npm(self):
        self.test_bitwise_and_ufunc(flags=no_pyobj_flags)

    def test_bitwise_or_ufunc(self, flags=enable_pyobj_flags):
        self.binary_int_ufunc_test(np.bitwise_or, flags=flags)

    def test_bitwise_or_ufunc_npm(self):
        self.test_bitwise_or_ufunc(flags=no_pyobj_flags)

    def test_bitwise_xor_ufunc(self, flags=enable_pyobj_flags):
        self.binary_int_ufunc_test(np.bitwise_xor, flags=flags)

    def test_bitwise_xor_ufunc_npm(self):
        self.test_bitwise_xor_ufunc(flags=no_pyobj_flags)

    def test_invert_ufunc(self, flags=enable_pyobj_flags):
        self.unary_int_ufunc_test(np.invert, flags=flags)

    def test_invert_ufunc_npm(self):
        self.test_invert_ufunc(flags=no_pyobj_flags)

    def test_bitwise_not_ufunc(self, flags=enable_pyobj_flags):
        self.unary_int_ufunc_test(np.bitwise_not, flags=flags)

    def test_bitwise_not_ufunc_npm(self):
        self.test_bitwise_not_ufunc(flags=no_pyobj_flags)


    # Note: there is no entry for left_shift and right_shift as this harness
    #       is not valid for them. This is so because left_shift and right
    #       shift implementation in NumPy has undefined behavior (in C-parlance)
    #       when the second argument is a negative (or bigger than the number
    #       of bits) value.
    #       Also, right_shift for negative first arguments also relies on
    #       implementation defined behavior, although numba warantees "sane"
    #       behavior (arithmetic shifts on signed integers, logic shifts on
    #       unsigned integers).

    ############################################################################
    # Comparison functions
    def test_greater_ufunc(self, flags=enable_pyobj_flags):
        self.binary_ufunc_test(np.greater, flags=flags)

    def test_greater_ufunc_npm(self):
        self.test_greater_ufunc(flags=no_pyobj_flags)

    def test_greater_equal_ufunc(self, flags=enable_pyobj_flags):
        self.binary_ufunc_test(np.greater_equal, flags=flags)

    def test_greater_equal_ufunc_npm(self):
        self.test_greater_equal_ufunc(flags=no_pyobj_flags)

    def test_less_ufunc(self, flags=enable_pyobj_flags):
        self.binary_ufunc_test(np.less, flags=flags)

    def test_less_ufunc_npm(self):
        self.test_less_ufunc(flags=no_pyobj_flags)

    def test_less_equal_ufunc(self, flags=enable_pyobj_flags):
        self.binary_ufunc_test(np.less_equal, flags=flags)

    def test_less_equal_ufunc_npm(self):
        self.test_less_equal_ufunc(flags=no_pyobj_flags)

    def test_not_equal_ufunc(self, flags=enable_pyobj_flags):
        self.binary_ufunc_test(np.not_equal, flags=flags)

    def test_not_equal_ufunc_npm(self):
        self.test_not_equal_ufunc(flags=no_pyobj_flags)

    def test_equal_ufunc(self, flags=enable_pyobj_flags):
        self.binary_ufunc_test(np.equal, flags=flags)

    def test_equal_ufunc_npm(self):
        self.test_equal_ufunc(flags=no_pyobj_flags)

    def test_logical_and_ufunc(self, flags=enable_pyobj_flags):
        self.binary_ufunc_test(np.logical_and, flags=flags)

    def test_logical_and_ufunc_npm(self):
        self.test_logical_and_ufunc(flags=no_pyobj_flags)

    def test_logical_or_ufunc(self, flags=enable_pyobj_flags):
        self.binary_ufunc_test(np.logical_or, flags=flags)

    def test_logical_or_ufunc_npm(self):
        self.test_logical_or_ufunc(flags=no_pyobj_flags)

    def test_logical_xor_ufunc(self, flags=enable_pyobj_flags):
        self.binary_ufunc_test(np.logical_xor, flags=flags)

    def test_logical_xor_ufunc_npm(self):
        self.test_logical_xor_ufunc(flags=no_pyobj_flags)

    def test_logical_not_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.logical_not, flags=flags)

    def test_logical_not_ufunc_npm(self):
        self.test_logical_not_ufunc(flags=no_pyobj_flags)

    def test_maximum_ufunc(self, flags=enable_pyobj_flags):
        self.binary_ufunc_test(np.maximum, flags=flags)

    def test_maximum_ufunc_npm(self):
        self.test_maximum_ufunc(flags=no_pyobj_flags)

    def test_minimum_ufunc(self, flags=enable_pyobj_flags):
        self.binary_ufunc_test(np.minimum, flags=flags)

    def test_minimum_ufunc_npm(self):
        self.test_minimum_ufunc(flags=no_pyobj_flags)

    def test_fmax_ufunc(self, flags=enable_pyobj_flags):
        self.binary_ufunc_test(np.fmax, flags=flags)

    def test_fmax_ufunc_npm(self):
        self.test_fmax_ufunc(flags=no_pyobj_flags)

    def test_fmin_ufunc(self, flags=enable_pyobj_flags):
        self.binary_ufunc_test(np.fmin, flags=flags)

    def test_fmin_ufunc_npm(self):
        self.test_fmin_ufunc(flags=no_pyobj_flags)


    ############################################################################
    # Floating functions
    def test_isfinite_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.isfinite, flags=flags)

    def test_isfinite_ufunc_npm(self):
        self.test_isfinite_ufunc(flags=no_pyobj_flags)

    def test_isinf_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.isinf, flags=flags)

    def test_isinf_ufunc_npm(self):
        self.test_isinf_ufunc(flags=no_pyobj_flags)

    def test_isnan_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.isnan, flags=flags)

    def test_isnan_ufunc_npm(self):
        self.test_isnan_ufunc(flags=no_pyobj_flags)

    def test_signbit_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.signbit, flags=flags)

    def test_signbit_ufunc_npm(self):
        self.test_signbit_ufunc(flags=no_pyobj_flags)

    def test_copysign_ufunc(self, flags=enable_pyobj_flags):
        self.binary_ufunc_test(np.copysign, flags=flags, kinds='f')

    def test_copysign_ufunc_npm(self):
        self.test_copysign_ufunc(flags=no_pyobj_flags)

    def test_nextafter_ufunc(self, flags=enable_pyobj_flags):
        self.binary_ufunc_test(np.nextafter, flags=flags, kinds='f')

    def test_nextafter_ufunc_npm(self):
        self.test_nextafter_ufunc(flags=no_pyobj_flags)

    def test_modf_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.modf, flags=flags, kinds='f')

    @_unimplemented
    def test_modf_ufunc_npm(self):
        self.test_modf_ufunc(flags=no_pyobj_flags)

    # Note: there is no entry for ldexp as this harness isn't valid for this
    #       ufunc. this is so because ldexp requires heterogeneous inputs.
    #       However, this ufunc is tested by the TestLoopTypes test classes.

    def test_frexp_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.frexp, flags=flags, kinds='f')

    @_unimplemented
    def test_frexp_ufunc_npm(self):
        self.test_frexp_ufunc(flags=no_pyobj_flags)

    def test_floor_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.floor, flags=flags, kinds='f')

    def test_floor_ufunc_npm(self):
        self.test_floor_ufunc(flags=no_pyobj_flags)

    def test_ceil_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.ceil, flags=flags, kinds='f')

    def test_ceil_ufunc_npm(self):
        self.test_ceil_ufunc(flags=no_pyobj_flags)

    def test_trunc_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.trunc, flags=flags, kinds='f')

    def test_trunc_ufunc_npm(self):
        self.test_trunc_ufunc(flags=no_pyobj_flags)

    def test_spacing_ufunc(self, flags=enable_pyobj_flags):
        self.unary_ufunc_test(np.spacing, flags=flags, kinds='f')

    def test_spacing_ufunc_npm(self):
        self.test_spacing_ufunc(flags=no_pyobj_flags)

    ############################################################################
    # Other tests
    def test_binary_ufunc_performance(self):

        pyfunc = _make_binary_ufunc_usecase(np.add)
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

    def binary_ufunc_mixed_types_test(self, ufunc, flags=enable_pyobj_flags):
        ufunc_name = ufunc.__name__
        ufunc = _make_binary_ufunc_usecase(ufunc)
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
        if not numpy_support.strict_ufunc_typing:
            self.binary_ufunc_mixed_types_test(np.divide, flags=no_pyobj_flags)

    def test_broadcasting(self):

        # Test unary ufunc
        pyfunc = _make_unary_ufunc_usecase(np.negative)

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
        pyfunc = _make_binary_ufunc_usecase(np.add)

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

    def test_implicit_output_npm(self):
        with self.assertRaises(TypeError):
            def myadd(a0, a1):
                return np.add(a0, a1)
            arr_ty = types.Array(types.uint64, 1, 'C')
            cr = compile_isolated(myadd, (arr_ty, arr_ty),
                                  flags=no_pyobj_flags)

    def test_broadcast_implicit_output_npm_nrt(self):
        def pyfunc(a0, a1):
            return np.add(a0, a1)

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

            cr = self.cache.compile(pyfunc, (input1_type, input2_type),
                                    flags=enable_nrt_flags)
            cfunc = cr.entry_point

            expected = np.add(x, y)
            result = cfunc(x, y)
            np.testing.assert_array_equal(expected, result)

    def test_implicit_output_layout_binary(self):
        def pyfunc(a0, a1):
            return np.add(a0, a1)

        # C layout
        X = np.linspace(0, 1, 20).reshape(4, 5)
        # F layout
        Y = np.array(X, order='F')
        # A layout
        Z = X.reshape(5, 4).T[0]

        Xty = typeof(X)
        assert X.flags.c_contiguous and Xty.layout == 'C'
        Yty = typeof(Y)
        assert Y.flags.f_contiguous and Yty.layout == 'F'
        Zty = typeof(Z)
        assert Zty.layout == 'A'
        assert not Z.flags.c_contiguous
        assert not Z.flags.f_contiguous

        testcases = list(itertools.permutations([X, Y, Z], 2))
        testcases += [(X, X)]
        testcases += [(Y, Y)]
        testcases += [(Z, Z)]

        for arg0, arg1 in testcases:
            cr = self.cache.compile(pyfunc, (typeof(arg0), typeof(arg1)),
                                    flags=enable_nrt_flags)
            expected = pyfunc(arg0, arg1)
            result = cr.entry_point(arg0, arg1)

            self.assertEqual(expected.flags.c_contiguous,
                             result.flags.c_contiguous)
            self.assertEqual(expected.flags.f_contiguous,
                             result.flags.f_contiguous)
            np.testing.assert_array_equal(expected, result)

    def test_implicit_output_layout_unary(self):
        def pyfunc(a0):
            return np.sqrt(a0)

        # C layout
        X = np.linspace(0, 1, 20).reshape(4, 5)
        # F layout
        Y = np.array(X, order='F')
        # A layout
        Z = X.reshape(5, 4).T[0]

        Xty = typeof(X)
        assert X.flags.c_contiguous and Xty.layout == 'C'
        Yty = typeof(Y)
        assert Y.flags.f_contiguous and Yty.layout == 'F'
        Zty = typeof(Z)
        assert Zty.layout == 'A'
        assert not Z.flags.c_contiguous
        assert not Z.flags.f_contiguous

        for arg0 in [X, Y, Z]:
            cr = self.cache.compile(pyfunc, (typeof(arg0),),
                                    flags=enable_nrt_flags)
            expected = pyfunc(arg0)
            result = cr.entry_point(arg0)

            self.assertEqual(expected.flags.c_contiguous,
                             result.flags.c_contiguous)
            self.assertEqual(expected.flags.f_contiguous,
                             result.flags.f_contiguous)
            np.testing.assert_array_equal(expected, result)



class TestArrayOperators(BaseUFuncTest, TestCase):

    def unary_op_test(self, operator, flags=enable_nrt_flags,
                      skip_inputs=[], additional_inputs=[],
                      int_output_type=None, float_output_type=None):
        operator_func = _make_unary_ufunc_op_usecase(operator)
        inputs = list(self.inputs)
        inputs.extend(additional_inputs)
        pyfunc = operator_func
        for input_tuple in inputs:
            input_operand, input_type = input_tuple

            if ((input_type in skip_inputs) or
                (not isinstance(input_type, types.Array))):
                continue

            cr = self.cache.compile(pyfunc, (input_type,),
                                    flags=flags)
            cfunc = cr.entry_point
            expected = pyfunc(input_operand)
            result = cfunc(input_operand)
            np.testing.assert_array_almost_equal(expected, result)

    def binary_op_test(self, operator, flags=enable_nrt_flags,
                       skip_inputs=[], additional_inputs=[],
                       int_output_type=None, float_output_type=None,
                       positive_rhs=False):
        operator_func = _make_binary_ufunc_op_usecase(operator)
        inputs = list(self.inputs)
        inputs.extend(additional_inputs)
        pyfunc = operator_func
        for input_tuple in inputs:
            input_operand1, input_type = input_tuple
            input_dtype = numpy_support.as_dtype(
                getattr(input_type, "dtype", input_type))
            input_type1 = input_type

            if input_type in skip_inputs:
                continue

            if positive_rhs:
                zero = np.zeros(1, dtype=input_dtype)[0]
            # If we only use two scalars, the code generator will not
            # select the ufunctionalized operator, so we mix it up.
            if isinstance(input_type, types.Array):
                input_operand0 = input_operand1
                input_type0 = input_type
                if positive_rhs and np.any(input_operand1 < zero):
                    continue
            else:
                input_operand0 = (np.random.random(10) * 100).astype(
                    input_dtype)
                input_type0 = typeof(input_operand0)
                if positive_rhs and input_operand1 < zero:
                    continue

            cr = self.cache.compile(pyfunc, (input_type0, input_type1),
                                    flags=flags)
            cfunc = cr.entry_point
            expected = pyfunc(input_operand0, input_operand1)
            result = cfunc(input_operand0, input_operand1)
            np.testing.assert_array_almost_equal(expected, result)

    def binary_int_op_test(self, *args, **kws):
        if 'skip_inputs' not in kws:
            kws['skip_inputs'] = []
        kws['skip_inputs'].extend([
            types.float32, types.float64,
            types.Array(types.float32, 1, 'C'),
            types.Array(types.float64, 1, 'C')])
        return self.binary_op_test(*args, **kws)

    def _make_arrays(self, dtypes):
        for dtype in dtypes:
            yield np.linspace(0, 5, 3).astype(dtype)
            yield np.linspace(1, 6, 3).astype(dtype)

    def inplace_op_test(self, operator, lhs_values, rhs_values,
                        lhs_dtypes, rhs_dtypes):
        operator_func = _make_inplace_ufunc_op_usecase(operator)
        pyfunc = operator_func

        # The left operand can only be an array, while the right operand
        # can be either an array or a scalar
        lhs_inputs = [np.array(lhs_values, dtype=dtype)
                      for dtype in lhs_dtypes]

        rhs_arrays = [np.array(rhs_values, dtype=dtype)
                      for dtype in rhs_dtypes]
        rhs_scalars = [dtype(v) for v in rhs_values for dtype in rhs_dtypes]
        rhs_inputs = rhs_arrays + rhs_scalars

        for lhs, rhs in itertools.product(lhs_inputs, rhs_inputs):
            lhs_type = typeof(lhs)
            rhs_type = typeof(rhs)
            cr = self.cache.compile(pyfunc, (lhs_type, rhs_type),
                                    flags=no_pyobj_flags)
            cfunc = cr.entry_point
            expected = lhs.copy()
            pyfunc(expected, rhs)
            got = lhs.copy()
            cfunc(got, rhs)
            self.assertPreciseEqual(got, expected)

    def inplace_float_op_test(self, operator, lhs_values, rhs_values):
        # Also accept integer inputs for the right operand (they should
        # be converted to float).
        return self.inplace_op_test(operator, lhs_values, rhs_values,
                                    (np.float32, np.float64),
                                    (np.float32, np.float64, np.int64))

    def inplace_int_op_test(self, operator, lhs_values, rhs_values):
        return self.inplace_op_test(operator, lhs_values, rhs_values,
                                    (np.int16, np.int32, np.int64),
                                    (np.int16, np.uint32))

    # ____________________________________________________________
    # Unary operators

    def test_unary_positive_array_op(self):
        self.unary_op_test('+')

    def test_unary_negative_array_op(self):
        self.unary_op_test('-')

    def test_unary_invert_array_op(self):
        self.unary_op_test('~', skip_inputs=[
            types.float32, types.float64,
            types.Array(types.float32, 1, 'C'),
            types.Array(types.float64, 1, 'C')])

    # ____________________________________________________________
    # Inplace operators

    def test_inplace_add(self):
        self.inplace_float_op_test('+=', [-1, 1.5, 3], [-5, 0, 2.5])

    def test_inplace_sub(self):
        self.inplace_float_op_test('-=', [-1, 1.5, 3], [-5, 0, 2.5])

    def test_inplace_mul(self):
        self.inplace_float_op_test('*=', [-1, 1.5, 3], [-5, 0, 2.5])

    def test_inplace_floordiv(self):
        self.inplace_float_op_test('//=', [-1, 1.5, 3], [-5, 0, 2.5])

    def test_inplace_div(self):
        self.inplace_float_op_test('/=', [-1, 1.5, 3], [-5, 0, 2.5])

    def test_inplace_remainder(self):
        self.inplace_float_op_test('%=', [-1, 1.5, 3], [-5, 2, 2.5])

    def test_inplace_pow(self):
        self.inplace_float_op_test('**=', [-1, 1.5, 3], [-5, 2, 2.5])

    def test_inplace_and(self):
        self.inplace_int_op_test('&=', [0, 1, 2, 3, 51], [0, 13, 16, 42, 255])

    def test_inplace_or(self):
        self.inplace_int_op_test('|=', [0, 1, 2, 3, 51], [0, 13, 16, 42, 255])

    def test_inplace_xor(self):
        self.inplace_int_op_test('^=', [0, 1, 2, 3, 51], [0, 13, 16, 42, 255])

    def test_inplace_lshift(self):
        self.inplace_int_op_test('<<=', [0, 5, -10, -51], [0, 1, 4, 14])

    def test_inplace_rshift(self):
        self.inplace_int_op_test('>>=', [0, 5, -10, -51], [0, 1, 4, 14])

    def test_unary_positive_array_op(self):
        '''
        Verify that the unary positive operator copies values, and doesn't
        just alias to the input array (mirrors normal Numpy/Python
        interaction behavior).
        '''
        # Test originally from @gmarkall
        def f(a1):
            a2 = +a1
            a1[0] = 3
            a2[1] = 4
            return a2

        a1 = np.zeros(10)
        a2 = f(a1)
        self.assertTrue(a1[0] != a2[0] and a1[1] != a2[1])
        a3 = np.zeros(10)
        a4 = njit(f)(a3)
        self.assertTrue(a3[0] != a4[0] and a3[1] != a4[1])
        np.testing.assert_array_equal(a1, a3)
        np.testing.assert_array_equal(a2, a4)

    # ____________________________________________________________
    # Binary operators

    def test_add_array_op(self):
        self.binary_op_test('+')

    def test_subtract_array_op(self):
        self.binary_op_test('-')

    def test_multiply_array_op(self):
        self.binary_op_test('*')

    def test_divide_array_op(self):
        int_out_type = None
        if PYVERSION >= (3, 0):
            int_out_type = types.float64
        self.binary_op_test('/', int_output_type=int_out_type)

    def test_floor_divide_array_op(self):
        self.binary_op_test('//')

    def test_remainder_array_op(self):
        self.binary_op_test('%')

    def test_power_array_op(self):
        self.binary_op_test('**')

    def test_left_shift_array_op(self):
        self.binary_int_op_test('<<', positive_rhs=True)

    def test_right_shift_array_op(self):
        self.binary_int_op_test('>>', positive_rhs=True)

    def test_bitwise_and_array_op(self):
        self.binary_int_op_test('&')

    def test_bitwise_or_array_op(self):
        self.binary_int_op_test('|')

    def test_bitwise_xor_array_op(self):
        self.binary_int_op_test('^')

    def test_equal_array_op(self):
        self.binary_op_test('==')

    def test_greater_array_op(self):
        self.binary_op_test('>')

    def test_greater_equal_array_op(self):
        self.binary_op_test('>=')

    def test_less_array_op(self):
        self.binary_op_test('<')

    def test_less_equal_array_op(self):
        self.binary_op_test('<=')

    def test_not_equal_array_op(self):
        self.binary_op_test('!=')


class TestScalarUFuncs(TestCase):
    """check the machinery of ufuncs works when the result is an scalar.
    These are not exhaustive because:
    - the machinery to support this case is the same for all the functions of a
      given arity.
    - the result of the inner function itself is already tested in TestUFuncs

    This class tests regular uses. A subclass tests the no python backend.
    """

    _compile_flags = enable_pyobj_flags

    def run_ufunc(self, pyfunc, arg_types, arg_values):
        for tyargs, args in zip(arg_types, arg_values):
            cr = compile_isolated(pyfunc, tyargs, flags=self._compile_flags)
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


    def test_scalar_unary_ufunc(self):
        def _func(x):
            return np.sqrt(x)

        vals = [(2,), (2,), (1,), (2,), (.1,), (.2,)]
        tys = [(types.int32,), (types.uint32,),
               (types.int64,), (types.uint64,),
               (types.float32,), (types.float64,)]
        self.run_ufunc(_func, tys, vals)


    def test_scalar_binary_uniform_ufunc(self):
        def _func(x,y):
            return np.add(x,y)

        vals = [2, 2, 1, 2, .1, .2]
        tys = [types.int32, types.uint32,
               types.int64, types.uint64, types.float32, types.float64]
        self.run_ufunc(_func, zip(tys, tys), zip(vals, vals))


    def test_scalar_binary_mixed_ufunc(self, flags=enable_pyobj_flags):
        def _func(x,y):
            return np.add(x,y)

        vals = [2, 2, 1, 2, .1, .2]
        tys = [types.int32, types.uint32,
               types.int64, types.uint64,
               types.float32, types.float64]
        self.run_ufunc(_func, itertools.product(tys, tys),
                       itertools.product(vals, vals))


class TestScalarUFuncsNoPython(TestScalarUFuncs):
    """Same tests as TestScalarUFuncs, but forcing no python mode"""
    _compile_flags = no_pyobj_flags


class TestUfuncIssues(TestCase):

    def test_issue_651(self):
        # Exercise the code path to make sure this does not fail
        @vectorize(["(float64,float64)"])
        def foo(x1, x2):
            return np.add(x1, x2) + np.add(x1, x2)

        a = np.arange(10, dtype='f8')
        b = np.arange(10, dtype='f8')
        self.assertTrue(np.all(foo(a, b) == (a + b) + (a + b)))

    def test_issue_713(self):
        def foo(x,y):
            return np.floor_divide(x,y)

        cr = compile_isolated(foo, [types.complex128, types.complex128])
        self.assertEqual(foo(1j, 1j), cr.entry_point(1j, 1j))


class _TestLoopTypes(TestCase):
    """Test code generation for the different loop types defined by ufunc.

    This class tests the ufuncs without forcing no-python mode. Subclasses
    of this class tweak it so they tests no-python mode support for the
    different ufuncs.

    This test relies on class variables to configure the test. Subclasses
    of this class can just override some of these variables to check other
    ufuncs in a different compilation context. The variables supported are:

    _funcs: the ufuncs to test
    _compile_flags: compilation flags to use (to force nopython mode)
    _skip_types: letter types that force skipping the loop when testing
                 if present in the NumPy ufunc signature.
    _supported_types: only test loops where all the types in the loop
                      signature are in this collection. If unset, all.

    Note that both, _skip_types and _supported_types must be met for a loop
    to be tested.

    The NumPy ufunc signature has a form like 'ff->f' (for a binary ufunc
    loop taking 2 floats and resulting in a float). In a NumPy ufunc object
    you can get a list of supported signatures by accessing the attribute
    'types'.
    """

    _ufuncs = all_ufuncs[:]
    # Have their own test classes
    _ufuncs.remove(np.left_shift)
    _ufuncs.remove(np.right_shift)
    _ufuncs.remove(np.reciprocal)
    _ufuncs.remove(np.power)
    _compile_flags = enable_pyobj_flags
    _skip_types = 'OegG'

    # Skip datetime64 'M' and timedelta64 'm' on numpy 1.6
    if is_on_numpy_16:
        _skip_types += 'Mm'

    _ulps = {('arccos', 'F'): 2,
             ('tanh', 'F'): 2,
             }

    def _arg_for_type(self, a_letter_type, index=0):
        """return a suitable array argument for testing the letter type"""
        if a_letter_type in 'bhilq':
            # an integral
            return np.array([1, 4, 0, -2], dtype=a_letter_type)
        if a_letter_type in 'BHILQ':
            return np.array([1, 2, 4, 0], dtype=a_letter_type)
        elif a_letter_type in '?':
            # a boolean
            return np.array([True, False, False, True], dtype=a_letter_type)
        elif a_letter_type[0] == 'm':
            # timedelta64
            if len(a_letter_type) == 1:
                a_letter_type = 'm8[D]'
            return np.array([2, -3, 'NaT', 0], dtype=a_letter_type)
        elif a_letter_type[0] == 'M':
            # datetime64
            if len(a_letter_type) == 1:
                a_letter_type = 'M8[D]'
            return np.array(['Nat', 1, 25, 0], dtype=a_letter_type)
        elif a_letter_type in 'fd':
            # floating point
            return np.array([1.5, -3.5, 0.0, float('nan')], dtype=a_letter_type)
        elif a_letter_type in 'FD':
            # complex
            return np.array([-1.0j, 1.5 + 1.5j, 1j * float('nan'), 0j],
                            dtype=a_letter_type)
        else:
            raise RuntimeError("type %r not understood" % (a_letter_type,))

    def _check_loop(self, fn, ufunc, loop):
        # the letter types for the args
        letter_types = loop[:ufunc.nin] + loop[-ufunc.nout:]

        # ignore the loops containing an object argument. They will always
        # fail in no python mode. Usually the last loop in ufuncs is an all
        # object fallback
        supported_types = getattr(self, '_supported_types', [])
        if (supported_types and
            any(l not in supported_types for l in letter_types)):
            return
        skip_types = getattr(self, '_skip_types', [])
        if any(l in skip_types for l in letter_types):
            return
        # if the test case requires some types to be present, skip loops
        # not involving any of those types.
        required_types = getattr(self, '_required_types', [])
        if required_types and not any(l in letter_types
                                      for l in required_types):
            return

        self._check_ufunc_with_dtypes(fn, ufunc, letter_types)

    def _check_ufunc_with_dtypes(self, fn, ufunc, dtypes):
        arg_dty = [np.dtype(t) for t in dtypes]
        arg_nbty = [types.Array(from_dtype(t), 1, 'C') for t in arg_dty]
        cr = compile_isolated(fn, arg_nbty, flags=self._compile_flags)

        # Ensure a good mix of input values
        c_args = [self._arg_for_type(t, index=index).repeat(2)
                  for index, t in enumerate(dtypes)]
        for arr in c_args:
            self.random.shuffle(arr)
        py_args = [a.copy() for a in c_args]

        cr.entry_point(*c_args)
        fn(*py_args)

        # Check each array (including inputs, to ensure they weren't
        # mutated).
        for c_arg, py_arg in zip(c_args, py_args):
            typechar = c_arg.dtype.char
            ulps = self._ulps.get((ufunc.__name__, typechar), 1)
            prec = 'single' if typechar in 'fF' else 'exact'
            prec = 'double' if typechar in 'dD' else prec
            msg = '\n'.join(["ufunc '{0}' arrays differ ({1}):",
                             "args: {2}", "expected {3}", "got {4}"])
            msg = msg.format(ufunc.__name__, c_args, prec, py_arg, c_arg)
            self.assertPreciseEqual(py_arg, c_arg, prec=prec, msg=msg,
                                    ulps=ulps)

    @classmethod
    def _check_ufunc_loops(cls, ufunc):
        for loop in ufunc.types:
            cls._inject_test(ufunc, loop)

    @classmethod
    def _inject_test(cls, ufunc, loop):
        def test_template(self):
            fn = _make_ufunc_usecase(ufunc)
            self._check_loop(fn, ufunc, loop)
        setattr(cls, "test_{0}_{1}".format(ufunc.__name__,
                                           loop.replace('->', '_')),
                test_template)

    @classmethod
    def autogenerate(cls):
        for ufunc in cls._ufuncs:
            cls._check_ufunc_loops(ufunc)


class TestLoopTypes(_TestLoopTypes):
    pass

TestLoopTypes.autogenerate()


class TestLoopTypesIntNoPython(_TestLoopTypes):
    _compile_flags = no_pyobj_flags
    _ufuncs = supported_ufuncs[:]
    # reciprocal and power need a special test due to issue #757
    _ufuncs.remove(np.power)
    _ufuncs.remove(np.reciprocal)
    _ufuncs.remove(np.left_shift) # has its own test class
    _ufuncs.remove(np.right_shift) # has its own test class
    _required_types = '?bBhHiIlLqQ'
    _skip_types = 'fdFDmMO' + _TestLoopTypes._skip_types

TestLoopTypesIntNoPython.autogenerate()

class TestLoopTypesReciprocalNoPython(_TestLoopTypes):
    _compile_flags = no_pyobj_flags
    _ufuncs = [np.reciprocal] # issue #757
    _required_types = 'bBhHiIlLqQfdFD'
    _skip_types = 'mMO' + _TestLoopTypes._skip_types

    def _arg_for_type(self, a_letter_type, index=0):
        res = super(self.__class__, self)._arg_for_type(a_letter_type,
                                                        index=index)
        if a_letter_type in 'bBhHiIlLqQ':
            # For integer reciprocal, avoid 0 as argument, as it triggers
            # undefined behavior that may differ in results from Numba
            # to the compiler used to compile NumPy.
            res[res == 0] = 42
        return res

TestLoopTypesReciprocalNoPython.autogenerate()

class TestLoopTypesPowerNoPython(_TestLoopTypes):
    _compile_flags = no_pyobj_flags
    _ufuncs = [np.power] # issue #757
    _required_types = 'bBhHiIlLqQfdFD'
    _skip_types = 'mMO' + _TestLoopTypes._skip_types

    def _arg_for_type(self, a_letter_type, index=0):
        res = super(self.__class__, self)._arg_for_type(a_letter_type,
                                                        index=index)
        if a_letter_type in 'bBhHiIlLqQ' and index == 1:
            # For integer power, avoid a negative exponent, as it triggers
            # undefined behavior that may differ in results from Numba
            # to the compiler used to compile NumPy
            res[res < 0] = 3
        return res

TestLoopTypesPowerNoPython.autogenerate()

class TestLoopTypesIntLeftShiftNoPython(_TestLoopTypes):
    _compile_flags = no_pyobj_flags
    _ufuncs = [np.left_shift]
    _required_types = 'bBhHiIlLqQ'
    _skip_types = 'fdFDmMO' + _TestLoopTypes._skip_types

    def _arg_for_type(self, a_letter_type, index=0):
        res = super(self.__class__, self)._arg_for_type(a_letter_type,
                                                        index=index)
        # Shifting by a negative amount (argument with index 1) is undefined
        # behavior in C. It is also undefined behavior in numba. In the same
        # sense, it is also undefined behavior when the shift amount is larger
        # than the number of bits in the shifted integer.
        # To avoid problems in the test, the values are clamped (clipped) so
        # that 0 <= shift_amount < bitcount(shifted_integer)
        if index == 1:
            bit_count = res.dtype.itemsize * 8
            res = np.clip(res, 0, bit_count-1)
        return res

TestLoopTypesIntLeftShiftNoPython.autogenerate()

class TestLoopTypesIntRightShiftNoPython(_TestLoopTypes):
    _compile_flags = no_pyobj_flags
    _ufuncs = [np.right_shift]
    _required_types = 'bBhHiIlLqQ'
    _skip_types = 'fdFDmMO' + _TestLoopTypes._skip_types

    def _arg_for_type(self, a_letter_type, index=0):
        res = super(self.__class__, self)._arg_for_type(a_letter_type,
                                                        index=index)
        # Shifting by a negative amount (argument with index 1) is undefined
        # behavior in C. It is also undefined behavior in numba. In the same
        # sense, it is also undefined behavior when the shift amount is larger
        # than the number of bits in the shifted integer.
        # To avoid problems in the test, the values are clamped (clipped) so
        # that 0 <= shift_amount < bitcount(shifted_integer)
        if index == 1:
            bit_count = res.dtype.itemsize * 8
            res = np.clip(res, 0, bit_count-1)

        # Right shift has "implementation defined behavior" when the number
        # shifted is negative (in C). In numba, right shift for signed integers
        # is "arithmetic" while for unsigned integers is "logical".
        # This test compares against the NumPy implementation, that relies
        # on "implementation defined behavior", so the test could be a false
        # failure if the compiler used to compile NumPy doesn't follow the same
        # policy.
        # Hint: do not rely on right shifting negative numbers in NumPy.
        if index == 0:
            res = np.abs(res)
        return res

TestLoopTypesIntRightShiftNoPython.autogenerate()

class TestLoopTypesFloatNoPython(_TestLoopTypes):
    _compile_flags = no_pyobj_flags
    _ufuncs = supported_ufuncs[:]
    if iswindows:
        _ufuncs.remove(np.signbit) # TODO: fix issue #758
    _required_types = 'fd'
    _skip_types = 'FDmMO' + _TestLoopTypes._skip_types

TestLoopTypesFloatNoPython.autogenerate()

class TestLoopTypesComplexNoPython(_TestLoopTypes):
    _compile_flags = no_pyobj_flags
    _ufuncs = supported_ufuncs[:]

    # Test complex types
    # Every loop containing a complex argument must be tested
    _required_types = 'FD'
    _skip_types = 'mMO' + _TestLoopTypes._skip_types

TestLoopTypesComplexNoPython.autogenerate()

@skip_on_numpy_16
class TestLoopTypesDatetimeNoPython(_TestLoopTypes):
    _compile_flags = no_pyobj_flags
    _ufuncs = supported_ufuncs[:]

    # NOTE: the full list of ufuncs supporting datetime64 and timedelta64
    # types in Numpy is:
    # ['absolute', 'add', 'divide', 'equal', 'floor_divide', 'fmax', 'fmin',
    #  'greater', 'greater_equal', 'less', 'less_equal', 'maximum',
    #  'minimum', 'multiply', 'negative', 'not_equal', 'sign', 'subtract',
    #  'true_divide']

    # Test datetime64 and timedelta64 types.
    _required_types = 'mM'

    # Test various units combinations (TestLoopTypes is only able to test
    # homogeneous units).

    def test_add(self):
        ufunc = np.add
        fn = _make_ufunc_usecase(ufunc)
        # heterogenous inputs
        self._check_ufunc_with_dtypes(fn, ufunc, ['m8[s]', 'm8[m]', 'm8[s]'])
        self._check_ufunc_with_dtypes(fn, ufunc, ['m8[m]', 'm8[s]', 'm8[s]'])
        if not numpy_support.strict_ufunc_typing:
            self._check_ufunc_with_dtypes(fn, ufunc, ['m8[m]', 'm8', 'm8[m]'])
            self._check_ufunc_with_dtypes(fn, ufunc, ['m8', 'm8[m]', 'm8[m]'])
        # heterogenous inputs, scaled output
        self._check_ufunc_with_dtypes(fn, ufunc, ['m8[s]', 'm8[m]', 'm8[ms]'])
        self._check_ufunc_with_dtypes(fn, ufunc, ['m8[m]', 'm8[s]', 'm8[ms]'])
        # Cannot upscale result (Numpy would accept this)
        with self.assertRaises(LoweringError):
            self._check_ufunc_with_dtypes(fn, ufunc, ['m8[m]', 'm8[s]', 'm8[m]'])

    def test_subtract(self):
        ufunc = np.subtract
        fn = _make_ufunc_usecase(ufunc)
        # heterogenous inputs
        self._check_ufunc_with_dtypes(fn, ufunc, ['M8[s]', 'M8[m]', 'm8[s]'])
        self._check_ufunc_with_dtypes(fn, ufunc, ['M8[m]', 'M8[s]', 'm8[s]'])
        # heterogenous inputs, scaled output
        self._check_ufunc_with_dtypes(fn, ufunc, ['M8[s]', 'M8[m]', 'm8[ms]'])
        self._check_ufunc_with_dtypes(fn, ufunc, ['M8[m]', 'M8[s]', 'm8[ms]'])
        # Cannot upscale result (Numpy would accept this)
        with self.assertRaises(LoweringError):
            self._check_ufunc_with_dtypes(fn, ufunc, ['M8[m]', 'M8[s]', 'm8[m]'])

    def test_multiply(self):
        ufunc = np.multiply
        fn = _make_ufunc_usecase(ufunc)
        # scaled output
        self._check_ufunc_with_dtypes(fn, ufunc, ['m8[s]', 'q', 'm8[us]'])
        self._check_ufunc_with_dtypes(fn, ufunc, ['q', 'm8[s]', 'm8[us]'])
        # Cannot upscale result (Numpy would accept this)
        with self.assertRaises(LoweringError):
            self._check_ufunc_with_dtypes(fn, ufunc, ['m8[s]', 'q', 'm8[m]'])

    def test_true_divide(self):
        ufunc = np.true_divide
        fn = _make_ufunc_usecase(ufunc)
        # heterogenous inputs
        self._check_ufunc_with_dtypes(fn, ufunc, ['m8[m]', 'm8[s]', 'd'])
        self._check_ufunc_with_dtypes(fn, ufunc, ['m8[s]', 'm8[m]', 'd'])
        # scaled output
        self._check_ufunc_with_dtypes(fn, ufunc, ['m8[m]', 'q', 'm8[s]'])
        self._check_ufunc_with_dtypes(fn, ufunc, ['m8[m]', 'd', 'm8[s]'])
        # Cannot upscale result (Numpy would accept this)
        with self.assertRaises(LoweringError):
            self._check_ufunc_with_dtypes(fn, ufunc, ['m8[s]', 'q', 'm8[m]'])

    def test_floor_divide(self):
        ufunc = np.floor_divide
        fn = _make_ufunc_usecase(ufunc)
        # scaled output
        self._check_ufunc_with_dtypes(fn, ufunc, ['m8[m]', 'q', 'm8[s]'])
        self._check_ufunc_with_dtypes(fn, ufunc, ['m8[m]', 'd', 'm8[s]'])
        # Cannot upscale result (Numpy would accept this)
        with self.assertRaises(LoweringError):
            self._check_ufunc_with_dtypes(fn, ufunc, ['m8[s]', 'q', 'm8[m]'])

    def _check_comparison(self, ufunc):
        fn = _make_ufunc_usecase(ufunc)
        # timedelta
        self._check_ufunc_with_dtypes(fn, ufunc, ['m8[m]', 'm8[s]', '?'])
        self._check_ufunc_with_dtypes(fn, ufunc, ['m8[s]', 'm8[m]', '?'])
        if not numpy_support.strict_ufunc_typing:
            self._check_ufunc_with_dtypes(fn, ufunc, ['m8[m]', 'm8', '?'])
            self._check_ufunc_with_dtypes(fn, ufunc, ['m8', 'm8[m]', '?'])
        # datetime
        self._check_ufunc_with_dtypes(fn, ufunc, ['M8[m]', 'M8[s]', '?'])
        self._check_ufunc_with_dtypes(fn, ufunc, ['M8[s]', 'M8[m]', '?'])

    def test_comparisons(self):
        for ufunc in [np.equal, np.not_equal, np.less, np.less_equal,
                      np.greater, np.greater_equal]:
            self._check_comparison(ufunc)

TestLoopTypesDatetimeNoPython.autogenerate()

class TestUFuncBadArgsNoPython(TestCase):
    _compile_flags = no_pyobj_flags

    def test_missing_args(self):
        def func(x):
            """error: np.add requires two args"""
            result = np.add(x)
            return result

        self.assertRaises(TypingError, compile_isolated, func, [types.float64],
                          return_type=types.float64, flags=self._compile_flags)


    def test_too_many_args(self):
        def func(x, out, out2):
            """error: too many args"""
            result = np.add(x, x, out, out2)
            return result

        array_type = types.Array(types.float64, 1, 'C')
        self.assertRaises(TypingError, compile_isolated, func, [array_type] *3,
                          return_type=array_type, flags=self._compile_flags)

    def test_no_scalar_result_by_reference(self):
        def func(x):
            """error: scalar as a return value is not supported"""
            y = 0
            np.add(x, x, y)
        self.assertRaises(TypingError, compile_isolated, func, [types.float64],
                          return_type=types.float64, flags=self._compile_flags)



if __name__ == '__main__':
    unittest.main()
