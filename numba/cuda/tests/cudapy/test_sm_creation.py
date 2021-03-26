import numpy as np
from numba import cuda, float32, int32
from numba.core.errors import TypingError
from numba.cuda.testing import unittest, CUDATestCase
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


def udt_invalid_3(A):
    sa = cuda.shared.array(shape=(1, A[0]), dtype=float32)
    i = cuda.grid(1)
    A[i] = sa[i, 0]


class TestSharedMemoryCreation(CUDATestCase):
    def getarg(self):
        return np.array(100, dtype=np.float32, ndmin=1)

    def getarg2(self):
        return self.getarg().reshape(1,1)

    def test_global_constants(self):
        udt = cuda.jit((float32[:],))(udt_global_constants)
        udt[1, 1](self.getarg())

    def test_global_build_tuple(self):
        udt = cuda.jit((float32[:, :],))(udt_global_build_tuple)
        udt[1, 1](self.getarg2())

    @skip_on_cudasim('Simulator does not prohibit lists for shared array shape')
    def test_global_build_list(self):
        with self.assertRaises(TypingError) as raises:
            cuda.jit((float32[:, :],))(udt_global_build_list)

        self.assertIn("No implementation of function "
                      "Function(<function shared.array",
                      str(raises.exception))
        self.assertIn("found for signature:\n \n "
                      ">>> array(shape=list(int64)<iv=[5, 6]>, "
                      "dtype=class(float32)",
                      str(raises.exception))

    def test_global_constant_tuple(self):
        udt = cuda.jit((float32[:, :],))(udt_global_constant_tuple)
        udt[1, 1](self.getarg2())

    @skip_on_cudasim("Can't check for constants in simulator")
    def test_invalid_1(self):
        # Scalar shape cannot be a floating point value
        with self.assertRaises(TypingError) as raises:
            cuda.jit((float32[:],))(udt_invalid_1)

        self.assertIn("No implementation of function "
                      "Function(<function shared.array",
                      str(raises.exception))
        self.assertIn("found for signature:\n \n "
                      ">>> array(shape=float32, dtype=class(float32))",
                      str(raises.exception))

    @skip_on_cudasim("Can't check for constants in simulator")
    def test_invalid_2(self):
        # Tuple shape cannot contain a floating point value
        with self.assertRaises(TypingError) as raises:
            cuda.jit((float32[:, :],))(udt_invalid_2)

        self.assertIn("No implementation of function "
                      "Function(<function shared.array",
                      str(raises.exception))
        self.assertIn("found for signature:\n \n "
                      ">>> array(shape=Tuple(Literal[int](1), "
                      "array(float32, 1d, A)), dtype=class(float32))",
                      str(raises.exception))

    @skip_on_cudasim("Can't check for constants in simulator")
    def test_invalid_3(self):
        # Scalar shape must be literal
        with self.assertRaises(TypingError) as raises:
            cuda.jit((int32[:],))(udt_invalid_1)

        self.assertIn("No implementation of function "
                      "Function(<function shared.array",
                      str(raises.exception))
        self.assertIn("found for signature:\n \n "
                      ">>> array(shape=int32, dtype=class(float32))",
                      str(raises.exception))

    @skip_on_cudasim("Can't check for constants in simulator")
    def test_invalid_4(self):
        # Tuple shape must contain only literals
        with self.assertRaises(TypingError) as raises:
            cuda.jit((int32[:],))(udt_invalid_3)

        self.assertIn("No implementation of function "
                      "Function(<function shared.array",
                      str(raises.exception))
        self.assertIn("found for signature:\n \n "
                      ">>> array(shape=Tuple(Literal[int](1), int32), "
                      "dtype=class(float32))",
                      str(raises.exception))


if __name__ == '__main__':
    unittest.main()
