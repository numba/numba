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

    def test_sin(self):
        @hsa.jit
        def sin(dst, src):
            i = hsa.get_global_id(0)
            if i < dst.size:
                dst[i] = math.sin(src[i])

        these_types = [np.float32, np.float64]
        for dtype in these_types:
            src = np.arange(10, dtype=dtype)
            dst = np.zeros_like(src)

            sin[src.size, 1](dst, src)

            np.testing.assert_allclose(dst, np.sin(src))

    def test_cos(self):
        @hsa.jit
        def cos(dst, src):
            i = hsa.get_global_id(0)
            if i < dst.size:
                dst[i] = math.cos(src[i])

        these_types = [np.float32, np.float64]
        for dtype in these_types:
            src = np.arange(10, dtype=dtype)
            dst = np.zeros_like(src)

            cos[src.size, 1](dst, src)

            np.testing.assert_allclose(dst, np.cos(src), rtol=1e-6)

    def test_tan(self):
        @hsa.jit
        def tan(dst, src):
            i = hsa.get_global_id(0)
            if i < dst.size:
                dst[i] = math.tan(src[i])

        these_types = [np.float64, np.float32]
        for dtype in these_types:
            src = np.arange(10, dtype=dtype)
            dst = np.zeros_like(src)

            tan[src.size, 1](dst, src)

            np.testing.assert_allclose(dst, np.tan(src), rtol=1e-6)


    def test_trig(self):
        types = [np.float32, np.float64]
        funcs = [math.sin, math.cos, math.tan]

        for fn in funcs:
            @hsa.jit
            def test_fn(dst, src):
                i = hsa.get_global_id(0)
                if i < dst.size:
                    dst[i] = fn(src[i])

            for dtype in types:
                tol = self._get_tol(dtype)
                src = np.linspace(-np.pi, np.pi, 720, dtype=dtype)
                dst = np.zeros_like(src)
                test_fn[src.size, 1](dst, src)
                np.testing.assert_allclose(dst, getattr(np, fn.__name__)(src),
                                           rtol=tol, err_msg=fn.__name__)


    def test_trig_inv(self):
        types = [np.float32, np.float64]
        funcs = [(math.asin, np.arcsin),
                 (math.acos, np.arccos),
                 (math.atan, np.arctan)]

        for fn, np_fn in funcs:
            @hsa.jit
            def test_fn(dst, src):
                i = hsa.get_global_id(0)
                if i < dst.size:
                    dst[i] = fn(src[i])

            for dtype in types:
                tol = self._get_tol(dtype)
                src = np.linspace(-1.0, 1.0, 720, dtype=dtype)
                dst = np.zeros_like(src)
                test_fn[src.size, 1](dst, src)

                np.testing.assert_allclose(dst, np_fn(src), rtol=tol,
                                           err_msg=fn.__name__)



if __name__ == '__main__':
    unittest.main()
