import numpy as np
import math
from numba import cuda
from numba.types import float32, float64, int32, void
from numba.cuda.testing import unittest, CUDATestCase


def simple_frexp(aryx, aryexp, arg):
    aryx[0], aryexp[0] = math.frexp(arg)


def simple_ldexp(aryx, arg, exp):
    aryx[0] = math.ldexp(arg, exp)


class TestCudaFrexpLdexp(CUDATestCase):
    def template_test_frexp(self, nptype, nbtype):
        compiled = cuda.jit(void(nbtype[::1], int32[::1], nbtype))(simple_frexp)
        arg = nbtype(3.1415)
        aryx = np.zeros(1, dtype=nptype)
        aryexp = np.zeros(1, dtype=np.int32)
        compiled[1, 1](aryx, aryexp, arg)
        np.testing.assert_array_equal(aryx, nptype(0.785375))
        self.assertEquals(aryexp, 2)

        arg = nbtype(np.inf)
        compiled[1, 1](aryx, aryexp, arg)
        np.testing.assert_array_equal(aryx, nptype(np.inf))
        self.assertEquals(aryexp, 0)  # np.frexp gives -1

        arg = nbtype(np.nan)
        compiled[1, 1](aryx, aryexp, arg)
        np.testing.assert_array_equal(aryx, nptype(np.nan))
        self.assertEquals(aryexp, 0)  # np.frexp gives -1

    def template_test_ldexp(self, nptype, nbtype):
        compiled = cuda.jit(void(nbtype[::1], nbtype, int32))(simple_ldexp)
        arg = nbtype(0.785375)
        exp = int32(2)
        aryx = np.zeros(1, dtype=nptype)
        compiled[1, 1](aryx, arg, exp)
        np.testing.assert_array_equal(aryx, nptype(3.1415))

        arg = nbtype(np.inf)
        compiled[1, 1](aryx, arg, exp)
        np.testing.assert_array_equal(aryx, nptype(np.inf))

        arg = nbtype(np.nan)
        compiled[1, 1](aryx, arg, exp)
        np.testing.assert_array_equal(aryx, nptype(np.nan))

    def test_frexp_f4(self):
        self.template_test_frexp(np.float32, float32)

    def test_ldexp_f4(self):
        self.template_test_ldexp(np.float32, float32)

    def test_frexp_f8(self):
        self.template_test_frexp(np.float64, float64)

    def test_ldexp_f8(self):
        self.template_test_ldexp(np.float64, float64)


if __name__ == '__main__':
    unittest.main()
