import numpy as np
import math
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase


def simple_frexp(aryx, aryexp, arg):
    aryx[0], aryexp[0] = math.frexp(arg)


def simple_ldexp(aryx, arg, exp):
    aryx[0] = math.ldexp(arg, exp)


class TestCudaFrexpLdexp(CUDATestCase):
    def test_frexp_f4(self):
        compiled = cuda.jit("void(f4[:], int32[:], f4)")(simple_frexp)
        arg = 3.1415
        aryx = np.zeros(1, dtype=np.float32)
        aryexp = np.zeros(1, dtype=np.int32)
        compiled[1, 1](aryx, aryexp, arg)
        np.testing.assert_array_equal(aryx, np.float32(0.785375))
        self.assertEquals(aryexp, 2)

    def test_ldexp_f4(self):
        compiled = cuda.jit("void(f4[:], f4, int32)")(simple_ldexp)
        arg = 0.785375
        exp = 2
        aryx = np.zeros(1, dtype=np.float32)
        compiled[1, 1](aryx, arg, exp)
        np.testing.assert_array_equal(aryx, np.float32(3.1415))

    def test_frexp_f8(self):
        compiled = cuda.jit("void(f8[:], int32[:], f8)")(simple_frexp)
        arg = 0.017
        aryx = np.zeros(1, dtype=np.float64)
        aryexp = np.zeros(1, dtype=np.int32)
        compiled[1, 1](aryx, aryexp, arg)
        np.testing.assert_array_equal(aryx, np.float64(0.544))
        self.assertEquals(aryexp, -5)

    def test_ldexp_f8(self):
        compiled = cuda.jit("void(f8[:], f8, int32)")(simple_ldexp)
        arg = 0.544
        exp = -5
        aryx = np.zeros(1, dtype=np.float64)
        compiled[1, 1](aryx, arg, exp)
        np.testing.assert_array_equal(aryx, np.float64(0.017))


if __name__ == '__main__':
    unittest.main()
