import numpy as np
import math
from .support import testcase, main, assertTrue
from numbapro import cuda, float64, int8, int32


def cu_mat_power(A, power, power_A):
    y, x = cuda.grid(2)

    m, n = power_A.shape
    if x >= n or y >= m:
        return

    power_A[y,x] = math.pow(A[y,x], power)


def cu_mat_power_binop(A, power, power_A):
    y, x = cuda.grid(2)

    m, n = power_A.shape
    if x >= n or y >= m:
        return

    power_A[y,x] = A[y,x] ** power

@testcase
def test_powi():
    dec = cuda.jit(argtypes=[float64[:,:], int8, float64[:,:]])
    kernel = dec(cu_mat_power)

    power = 2
    A = np.arange(10, dtype=np.float64).reshape(2, 5)
    Aout = np.empty_like(A)
    kernel[1, A.shape](A, power, Aout)
    assertTrue(np.allclose(Aout, A ** power))

@testcase
def test_powi_binop():
    dec = cuda.jit(argtypes=[float64[:,:], int8, float64[:,:]])
    kernel = dec(cu_mat_power_binop)

    power = 2
    A = np.arange(10, dtype=np.float64).reshape(2, 5)
    Aout = np.empty_like(A)
    kernel[1, A.shape](A, power, Aout)
    assertTrue(np.allclose(Aout, A ** power))

if __name__ == '__main__':
    main()

