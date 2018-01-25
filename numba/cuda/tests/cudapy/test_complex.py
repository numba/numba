from __future__ import print_function, absolute_import, division

import cmath
import math
import itertools
import string
import sys
import textwrap

import numpy as np

from numba.cuda.testing import unittest, SerialMixin
from numba import cuda, types, utils, numpy_support
from numba.tests.support import TestCase, compile_function
from numba.tests.complex_usecases import *


def compile_scalar_func(pyfunc, argtypes, restype):
    # First compile a scalar device function
    assert not any(isinstance(tp, types.Array) for tp in argtypes)
    assert not isinstance(restype, types.Array)
    device_func = cuda.jit(restype(*argtypes), device=True)(pyfunc)

    kernel_types = [types.Array(tp, 1, "C")
                    for tp in [restype] + list(argtypes)]

    if len(argtypes) == 1:
        def kernel_func(out, a):
            i = cuda.grid(1)
            if i < out.shape[0]:
                out[i] = device_func(a[i])
    elif len(argtypes) == 2:
        def kernel_func(out, a, b):
            i = cuda.grid(1)
            if i < out.shape[0]:
                out[i] = device_func(a[i], b[i])
    else:
        assert 0

    kernel = cuda.jit(tuple(kernel_types))(kernel_func)

    def kernel_wrapper(values):
        n = len(values)
        inputs = [np.empty(n, dtype=numpy_support.as_dtype(tp))
                  for tp in argtypes]
        output = np.empty(n, dtype=numpy_support.as_dtype(restype))
        for i, vs in enumerate(values):
            for v, inp in zip(vs, inputs):
                inp[i] = v
        args = [output] + inputs
        kernel[int(math.ceil(n / 256)), 256](*args)
        return list(output)
    return kernel_wrapper


class BaseComplexTest(SerialMixin):

    def basic_values(self):
        reals = [-0.0, +0.0, 1, -1, +1.5, -3.5,
                 float('-inf'), float('+inf'), float('nan')]
        return [complex(x, y) for x, y in itertools.product(reals, reals)]

    def more_values(self):
        reals = [0.0, +0.0, 1, -1, -math.pi, +math.pi,
                 float('-inf'), float('+inf'), float('nan')]
        return [complex(x, y) for x, y in itertools.product(reals, reals)]

    def non_nan_values(self):
        reals = [-0.0, +0.0, 1, -1, -math.pi, +math.pi,
                 float('inf'), float('-inf')]
        return [complex(x, y) for x, y in itertools.product(reals, reals)]

    def run_func(self, pyfunc, sigs, values, ulps=1, ignore_sign_on_zero=False):
        for sig in sigs:
            if isinstance(sig, types.Type):
                sig = sig,
            if isinstance(sig, tuple):
                # Assume return type is the type of first argument
                sig = sig[0](*sig)
            prec = ('single'
                    if sig.args[0] in (types.float32, types.complex64)
                    else 'double')
            cudafunc = compile_scalar_func(pyfunc, sig.args, sig.return_type)
            ok_values = []
            expected_list = []
            for args in values:
                if not isinstance(args, (list, tuple)):
                    args = args,
                try:
                    expected_list.append(pyfunc(*args))
                    ok_values.append(args)
                except ValueError as e:
                    self.assertIn("math domain error", str(e))
                    continue
            got_list = cudafunc(ok_values)
            for got, expected, args in zip(got_list, expected_list, ok_values):
                msg = 'for input %r with prec %r' % (args, prec)
                self.assertPreciseEqual(got, expected, prec=prec,
                                        ulps=ulps,
                                        ignore_sign_on_zero=ignore_sign_on_zero,
                                        msg=msg)

    run_unary = run_func
    run_binary = run_func


class TestComplex(BaseComplexTest, TestCase):

    def check_real_image(self, pyfunc):
        values = self.basic_values()
        self.run_unary(pyfunc,
                       [tp.underlying_float(tp)
                        for tp in (types.complex64, types.complex128)],
                       values)

    def test_real(self):
        self.check_real_image(real_usecase)

    def test_imag(self):
        self.check_real_image(imag_usecase)

    def test_conjugate(self):
        pyfunc = conjugate_usecase
        values = self.basic_values()
        self.run_unary(pyfunc,
                       [types.complex64, types.complex128],
                       values)


class TestCMath(BaseComplexTest, TestCase):
    """
    Tests for cmath module support.
    """

    def check_predicate_func(self, pyfunc):
        self.run_unary(pyfunc,
                       [types.boolean(tp) for tp in (types.complex128, types.complex64)],
                       self.basic_values())

    def check_unary_func(self, pyfunc, ulps=1, values=None,
                         returns_float=False, ignore_sign_on_zero=False):
        if returns_float:
            def sig(tp):
                return tp.underlying_float(tp)
        else:
            def sig(tp):
                return tp(tp)
        self.run_unary(pyfunc, [sig(types.complex128)],
                       values or self.more_values(), ulps=ulps,
                       ignore_sign_on_zero=ignore_sign_on_zero)
        # Avoid discontinuities around pi when in single precision.
        self.run_unary(pyfunc, [sig(types.complex64)],
                       values or self.basic_values(), ulps=ulps,
                       ignore_sign_on_zero=ignore_sign_on_zero)

    # Conversions

    def test_phase(self):
        self.check_unary_func(phase_usecase, returns_float=True)

    def test_polar(self):
        self.check_unary_func(polar_as_complex_usecase)

    def test_rect(self):
        def do_test(tp, seed_values):
            values = [(z.real, z.imag) for z in seed_values
                      if not math.isinf(z.imag) or z.real == 0]
            float_type = tp.underlying_float
            self.run_binary(rect_usecase, [tp(float_type, float_type)],
                            values)
        do_test(types.complex128, self.more_values())
        # Avoid discontinuities around pi when in single precision.
        do_test(types.complex64, self.basic_values())

    # Classification

    def test_isnan(self):
        self.check_predicate_func(isnan_usecase)

    def test_isinf(self):
        self.check_predicate_func(isinf_usecase)

    @unittest.skipIf(utils.PYVERSION < (3, 2), "needs Python 3.2+")
    def test_isfinite(self):
        self.check_predicate_func(isfinite_usecase)

    # Power and logarithms

    def test_exp(self):
        self.check_unary_func(exp_usecase, ulps=2)

    def test_log(self):
        self.check_unary_func(log_usecase)

    def test_log_base(self):
        values = list(itertools.product(self.more_values(), self.more_values()))
        value_types = [(types.complex128, types.complex128),
                       (types.complex64, types.complex64)]
        self.run_binary(log_base_usecase, value_types, values,
                        ulps=3)

    def test_log10(self):
        self.check_unary_func(log10_usecase)

    def test_sqrt(self):
        self.check_unary_func(sqrt_usecase)

    # Trigonometric functions

    def test_acos(self):
        self.check_unary_func(acos_usecase, ulps=2)

    def test_asin(self):
        self.check_unary_func(asin_usecase, ulps=2)

    def test_atan(self):
        self.check_unary_func(atan_usecase, ulps=2,
                              values=self.non_nan_values())

    def test_cos(self):
        self.check_unary_func(cos_usecase, ulps=2)

    def test_sin(self):
        # See test_sinh.
        self.check_unary_func(sin_usecase, ulps=2)

    def test_tan(self):
        self.check_unary_func(tan_usecase, ulps=2,
                              ignore_sign_on_zero=True)

    # Hyperbolic functions

    def test_acosh(self):
        self.check_unary_func(acosh_usecase)

    def test_asinh(self):
        self.check_unary_func(asinh_usecase, ulps=2)

    def test_atanh(self):
        self.check_unary_func(atanh_usecase, ulps=2,
                              ignore_sign_on_zero=True)

    def test_cosh(self):
        self.check_unary_func(cosh_usecase, ulps=2)

    def test_sinh(self):
        self.check_unary_func(sinh_usecase, ulps=2)

    def test_tanh(self):
        self.check_unary_func(tanh_usecase, ulps=2,
                              ignore_sign_on_zero=True)


if __name__ == '__main__':
    unittest.main()
