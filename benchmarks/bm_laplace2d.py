from __future__ import absolute_import, print_function, division

import numpy as np
from numba import jit
from numba.utils import benchmark


def jacobi_relax_core(A, Anew):
    error = 0.0
    n = A.shape[0]
    m = A.shape[1]

    for j in range(1, n - 1):
        for i in range(1, m - 1):
            Anew[j, i] = 0.25 * ( A[j, i + 1] + A[j, i - 1] \
                                + A[j - 1, i] + A[j + 1, i])
            error = max(error, abs(Anew[j, i] - A[j, i]))
    return error


numba_jacobi_relax_core = jit("float64[:,::1], float64[:,::1]", nopython=True)\
                             (jacobi_relax_core)


def run(fn):
    NN = 1024
    NM = 1024

    A = np.zeros((NN, NM), dtype=np.float64)
    Anew = np.zeros((NN, NM), dtype=np.float64)

    n = NN
    m = NM
    iter_max = 10

    tol = 1.0e-6
    error = 1.0

    for j in range(n):
        A[j, 0] = 1.0
        Anew[j, 0] = 1.0

    it = 0

    while error > tol and it < iter_max:
        error = fn(A, Anew)

        # swap A and Anew
        tmp = A
        A = Anew
        Anew = tmp
        it += 1


def python_main():
    run(jacobi_relax_core)


def numba_main():
    run(numba_jacobi_relax_core)


if __name__ == '__main__':
    print(benchmark(python_main))
    print(benchmark(numba_main))