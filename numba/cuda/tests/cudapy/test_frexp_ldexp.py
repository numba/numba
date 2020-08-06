import numpy as np
import math
from numba import cuda
from numba.types import float32, float64, int32
from numba.cuda.testing import unittest, CUDATestCase


def simple_frexp(x, exp, arg):
    x[0], exp[0] = math.frexp(arg)


def simple_ldexp(val, arg, exp):
    val[0] = math.ldexp(arg, exp)


class TestCudaFrexpLdexp(CUDATestCase):
    def template_test_frexp_ldexp(self, value, npyty, npmty):
        frexp_kernel = cuda.jit((npmty[:], int32[:], npmty))(simple_frexp)

        arg = npyty(value)
        x = np.empty(1, dtype=npyty)
        xneg = np.empty_like(x)
        exp = np.empty_like(x, dtype=np.int32)
        npx, npexp = np.frexp(arg)

        frexp_kernel[1, 1](x, exp, arg)

        np.testing.assert_array_equal(x, npx)
        self.assertEquals(exp, npexp)

        frexp_kernel[1, 1](xneg, exp, -arg)

        np.testing.assert_array_equal(xneg, -npx)
        self.assertEquals(exp, npexp)

        ldexp_kernel = cuda.jit((npmty[:], npmty, int32))(simple_ldexp)

        val = np.empty(1, dtype=npyty)

        ldexp_kernel[1, 1](val, x.item(), exp.item())

        np.testing.assert_array_equal(val, np.ldexp(x, exp))
        np.testing.assert_array_equal(val, arg)  # roundtrip

        ldexp_kernel[1, 1](val, xneg.item(), exp.item())

        np.testing.assert_array_equal(val, np.ldexp(xneg, exp))
        np.testing.assert_array_equal(val, -arg)  # roundtrip

    def test_ldexp_frexp(self):
        for value in [0., 3.1e-10, 0.785375, 2., 1.3e10]:
            self.template_test_frexp_ldexp(value, np.float32, float32)
            self.template_test_frexp_ldexp(value, np.float64, float64)


if __name__ == '__main__':
    unittest.main()
