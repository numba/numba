from __future__ import print_function, division, absolute_import
import numpy as np
from numba import cuda, float32
from numba.cuda.testing import unittest
from numba.errors import MacroError
from numba.cuda.testing import skip_on_cudasim

GLOBAL_CONSTANT = 5
GLOBAL_CONSTANT_2 = 6
GLOBAL_CONSTANT_TUPLE = 5, 6


def udt_global_constants(A):
    sa = cuda.shared.array(shape=GLOBAL_CONSTANT, dtype=float32)
    i = cuda.grid(1)
    A[i] = sa[i]


def udt_global_build_tuple(A):
    sa = cuda.shared.array(shape=(GLOBAL_CONSTANT, GLOBAL_CONSTANT_2),
                           dtype=float32)
    i, j = cuda.grid(2)
    A[i, j] = sa[i, j]


def udt_global_build_list(A):
    sa = cuda.shared.array(shape=[GLOBAL_CONSTANT, GLOBAL_CONSTANT_2],
                           dtype=float32)
    i, j = cuda.grid(2)
    A[i, j] = sa[i, j]


def udt_global_constant_tuple(A):
    sa = cuda.shared.array(shape=GLOBAL_CONSTANT_TUPLE, dtype=float32)
    i, j = cuda.grid(2)
    A[i, j] = sa[i, j]


def udt_invalid_1(A):
    sa = cuda.shared.array(shape=A[0], dtype=float32)
    i = cuda.grid(1)
    A[i] = sa[i]


def udt_invalid_2(A):
    sa = cuda.shared.array(shape=(1, A[0]), dtype=float32)
    i, j = cuda.grid(2)
    A[i, j] = sa[i, j]


class TestMacro(unittest.TestCase):
    def getarg(self):
        return np.array(100, dtype=np.float32, ndmin=1)

    def getarg2(self):
        return self.getarg().reshape(1,1)

    def test_global_constants(self):
        udt = cuda.jit((float32[:],))(udt_global_constants)
        udt(self.getarg())

    def test_global_build_tuple(self):
        udt = cuda.jit((float32[:, :],))(udt_global_build_tuple)
        udt(self.getarg2())

    @skip_on_cudasim('Simulator does not perform macro expansion')
    def test_global_build_list(self):
        with self.assertRaises(MacroError) as raises:
            cuda.jit((float32[:, :],))(udt_global_build_list)

        self.assertIn("invalid type for shape; got {0}".format(list),
                      str(raises.exception))

    def test_global_constant_tuple(self):
        udt = cuda.jit((float32[:, :],))(udt_global_constant_tuple)
        udt(self.getarg2())

    @skip_on_cudasim("Can't check for constants in simulator")
    def test_invalid_1(self):
        with self.assertRaises(ValueError) as raises:
            cuda.jit((float32[:],))(udt_invalid_1)

        self.assertIn("Argument 'shape' must be a constant at",
                      str(raises.exception))

    @skip_on_cudasim("Can't check for constants in simulator")
    def test_invalid_2(self):
        with self.assertRaises(ValueError) as raises:
            cuda.jit((float32[:, :],))(udt_invalid_2)

        self.assertIn("Argument 'shape' must be a constant at",
                      str(raises.exception))


if __name__ == '__main__':
    unittest.main()
