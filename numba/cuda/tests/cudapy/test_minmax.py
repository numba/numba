from __future__ import print_function, absolute_import

import numpy as np

from numba import cuda
from numba.cuda.testing import unittest, SerialMixin, skip_on_cudasim


def builtin_max(A, B, C):
    i = cuda.grid(1)

    if i >= len(C):
        return

    C[i] = max(A[i], B[i])


def builtin_min(A, B, C):
    i = cuda.grid(1)

    if i >= len(C):
        return

    C[i] = min(A[i], B[i])


@skip_on_cudasim('Tests PTX emission')
class TestCudaMinMax(SerialMixin, unittest.TestCase):
    def test_max_f8(self, n=5):
        kernel = cuda.jit(builtin_max)

        c = np.zeros(n, dtype=np.float64)
        a = np.arange(n, dtype=np.float64) + .5
        b = np.full(n, fill_value=2, dtype=np.float64)

        kernel[1, c.shape](a, b, c)
        np.testing.assert_allclose(
            c,
            np.maximum(a, b))

        ptx = next(p for p in kernel.inspect_asm().values())
        assert 'max.f64' in ptx, ptx

    def test_min_f8(self, n=5):
        kernel = cuda.jit(builtin_min)

        c = np.zeros(n, dtype=np.float64)
        a = np.arange(n, dtype=np.float64) + .5
        b = np.full(n, fill_value=2, dtype=np.float64)

        kernel[1, c.shape](a, b, c)
        np.testing.assert_allclose(
            c,
            np.minimum(a, b))

        ptx = next(p for p in kernel.inspect_asm().values())
        assert 'min.f64' in ptx, ptx

    def test_max_f4(self, n=5):
        kernel = cuda.jit(builtin_max)

        c = np.zeros(n, dtype=np.float32)
        a = np.arange(n, dtype=np.float32) + .5
        b = np.full(n, fill_value=2, dtype=np.float32)

        kernel[1, c.shape](a, b, c)
        np.testing.assert_allclose(
            c,
            np.maximum(a, b))

        ptx = next(p for p in kernel.inspect_asm().values())
        assert 'max.f32' in ptx, ptx

    def test_min_f4(self, n=5):
        kernel = cuda.jit(builtin_min)

        c = np.zeros(n, dtype=np.float32)
        a = np.arange(n, dtype=np.float32) + .5
        b = np.full(n, fill_value=2, dtype=np.float32)

        kernel[1, c.shape](a, b, c)
        np.testing.assert_allclose(
            c,
            np.minimum(a, b))

        ptx = next(p for p in kernel.inspect_asm().values())
        assert 'min.f32' in ptx, ptx


if __name__ == '__main__':
    unittest.main()
