from __future__ import print_function, absolute_import

import numpy as np
import math

import numba.unittest_support as unittest
from numba import hsa
from numba import utils


class TestMath(unittest.TestCase):
    def _get_tol(self, math_fn, ty):
        """gets the tolerance for functions when the input is of type 'ty'"""

        low_res = {
            (math.gamma, np.float64): 1e-14,
            (math.lgamma, np.float64): 1e-13,
        }
        default = 1e-15 if ty == np.float64 else 1e-6
        return low_res.get((math_fn, ty), default)

    def _generic_test_unary(self, math_fn, npy_fn,
                            cases=None,
                            span=(-1., 1.), count=128,
                            types=(np.float32, np.float64)):

        @hsa.jit
        def fn(dst, src):
            i = hsa.get_global_id(0)
            if i < dst.size:
                dst[i] = math_fn(src[i])

        for dtype in types:
            if cases is None:
                src = np.linspace(span[0], span[1], count).astype(dtype)
            else:
                src = np.array(cases, dtype=dtype)

            dst = np.zeros_like(src)
            fn[src.size, 1](dst, src)
            np.testing.assert_allclose(dst, npy_fn(src),
                                       rtol=self._get_tol(math_fn, dtype),
                                       err_msg='{0} ({1})'.format(
                                           math_fn.__name__,
                                           dtype.__name__))

    def _generic_test_binary(self, math_fn, npy_fn,
                             cases=None,
                             span=(-1., 1., 1., -1.), count=128,
                             types=(np.float32, np.float64)):

        @hsa.jit
        def fn(dst, src1, src2):
            i = hsa.get_global_id(0)
            if i < dst.size:
                dst[i] = math_fn(src1[i], src2[i])

        for dtype in types:
            if cases is None:
                src1 = np.linspace(span[0], span[1], count).astype(dtype)
                src2 = np.linspace(span[2], span[3], count).astype(dtype)
            else:
                src1 = np.array(cases[0], dtype=dtype)
                src2 = np.array(cases[1], dtype=dtype)

            dst = np.zeros_like(src1)
            fn[dst.size, 1](dst, src1, src2)
            np.testing.assert_allclose(dst, npy_fn(src1, src2),
                                       rtol=self._get_tol(math_fn, dtype),
                                       err_msg='{0} ({1})'.format(
                                           math_fn.__name__,
                                           dtype.__name__))

    def test_trig(self):
        funcs = [math.sin, math.cos, math.tan]

        for fn in funcs:
            self._generic_test_unary(fn, getattr(np, fn.__name__),
                                     span=(-np.pi, np.pi))

    def test_trig_inv(self):
        funcs = [(math.asin, np.arcsin),
                 (math.acos, np.arccos),
                 (math.atan, np.arctan)]

        for fn, np_fn in funcs:
            self._generic_test_unary(fn, np_fn)

    def test_trigh(self):
        funcs = [math.sinh, math.cosh, math.tanh]
        for fn in funcs:
            self._generic_test_unary(fn, getattr(np, fn.__name__),
                                     span=(-4.0, 4.0))

    def test_trigh_inv(self):
        funcs = [(math.asinh, np.arcsinh, (-4, 4)),
                 (math.acosh, np.arccosh, (1, 9)),
                 (math.atanh, np.arctanh, (-0.9, 0.9))]

        for fn, np_fn, span in funcs:
            self._generic_test_unary(fn, np_fn, span=span)

    def test_classify(self):
        funcs = [math.isnan, math.isinf]
        cases = (float('nan'), float('inf'), float('-inf'), float('-nan'),
                 0, 3, -2)
        for fn in funcs:
            self._generic_test_unary(fn, getattr(np, fn.__name__),
                                     cases=cases)

    def test_floor_ceil(self):
        funcs = [math.ceil, math.floor]

        for fn in funcs:
            # cases with varied decimals
            self._generic_test_unary(fn, getattr(np, fn.__name__),
                                     span=(-1013.14, 843.21))
            # cases that include "exact" integers
            self._generic_test_unary(fn, getattr(np, fn.__name__),
                                     span=(-16, 16), count=129)

    def test_fabs(self):
        funcs = [math.fabs]
        for fn in funcs:
            self._generic_test_unary(fn, getattr(np, fn.__name__),
                                     span=(-63.3, 63.3))

    def test_unary_exp(self):
        funcs = [math.exp]
        for fn in funcs:
            self._generic_test_unary(fn, getattr(np, fn.__name__),
                                     span=(-30, 30))

    def test_unary_expm1(self):
        funcs = [math.expm1]
        for fn in funcs:
            self._generic_test_unary(fn, getattr(np, fn.__name__),
                                     span=(-30, 30))

    def test_sqrt(self):
        funcs = [math.sqrt]
        for fn in funcs:
            self._generic_test_unary(fn, getattr(np, fn.__name__),
                                     span=(0, 1000))

    def test_log(self):
        funcs = [math.log, math.log10, math.log1p]
        for fn in funcs:
            self._generic_test_unary(fn, getattr(np, fn.__name__),
                                     span=(0.1, 2500))

    def test_binaries(self):
        funcs = [math.copysign, math.fmod]
        for fn in funcs:
            self._generic_test_binary(fn, getattr(np, fn.__name__))

    def test_pow(self):
        funcs = [(math.pow, np.power)]
        for fn, npy_fn in funcs:
            self._generic_test_binary(fn, npy_fn)

    def test_atan2(self):
        funcs = [(math.atan2, np.arctan2)]
        for fn, npy_fn in funcs:
            self._generic_test_binary(fn, npy_fn)

    def test_erf(self):
        funcs = [math.erf, math.erfc]
        for fn in funcs:
            self._generic_test_unary(fn, np.vectorize(fn))

    @unittest.skipIf(True, "gamma() freezing computer on Carrizo (Kaveri may "
                           "crash the process but no harm to system)")
    def test_gamma(self):
        funcs = [math.gamma, math.lgamma]
        for fn in funcs:
            self._generic_test_unary(fn, np.vectorize(fn), span=(1e-4, 4.0))


if __name__ == '__main__':
    unittest.main()
