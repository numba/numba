from __future__ import print_function, absolute_import

import numpy as np
import math

import numba.unittest_support as unittest
from numba import hsa


class TestMath(unittest.TestCase):
    def _get_tol(self, ty):
        """gets the tolerance for functions when the input is of type 'ty'"""
        if ty == np.float64:
            return 1e-15
        else:
            return 1e-6

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
                src = np.linspace(span[0], span[1], count, dtype=dtype)
            else:
                src = np.array(cases, dtype=dtype)

            dst = np.zeros_like(src)
            fn[src.size, 1](dst, src)
            np.testing.assert_allclose(dst, npy_fn(src),
                                       rtol=self._get_tol(dtype),
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
                 (math.acosh, np.arccosh, ( 1, 9)),
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


if __name__ == '__main__':
    unittest.main()
