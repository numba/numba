import numpy as np

from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase


def noop(x):
    pass


class TestJitErrors(CUDATestCase):
    """
    Test compile-time errors with @jit.
    """

    def test_too_many_dims(self):
        kernfunc = cuda.jit(noop)

        with self.assertRaises(ValueError) as raises:
            kernfunc[(1, 2, 3, 4), (5, 6)]
        self.assertIn("griddim must be a sequence of 1, 2 or 3 integers, got [1, 2, 3, 4]",
                      str(raises.exception))

        with self.assertRaises(ValueError) as raises:
            kernfunc[(1, 2,), (3, 4, 5, 6)]
        self.assertIn("blockdim must be a sequence of 1, 2 or 3 integers, got [3, 4, 5, 6]",
                      str(raises.exception))

    def test_non_integral_dims(self):
        kernfunc = cuda.jit(noop)

        with self.assertRaises(TypeError) as raises:
            kernfunc[2.0, 3]
        self.assertIn("griddim must be a sequence of integers, got [2.0]",
                      str(raises.exception))

        with self.assertRaises(TypeError) as raises:
            kernfunc[2, 3.0]
        self.assertIn("blockdim must be a sequence of integers, got [3.0]",
                      str(raises.exception))

    def _test_unconfigured(self, kernfunc):
        with self.assertRaises(ValueError) as raises:
            kernfunc(0)
        self.assertIn("launch configuration was not specified",
                      str(raises.exception))

    def test_unconfigured_typed_cudakernel(self):
        kernfunc = cuda.jit("void(int32)")(noop)
        self._test_unconfigured(kernfunc)

    def test_unconfigured_untyped_cudakernel(self):
        kernfunc = cuda.jit(noop)
        self._test_unconfigured(kernfunc)


if __name__ == '__main__':
    unittest.main()
