# issue: #33
# Thanks to Stefan van der Walt

import numpy as np

from numba.decorators import jit
from numba import float64, int32

@jit(argtypes=[float64[:, :], float64[:, :], float64[:, :]])
def ndot(A, B, out):
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape

    # Take each row in A
    for i in range(rows_A):

        # And multiply by every column in B
        for j in range(cols_B):
            s = 0.0
            for k in range(cols_A):
                s = s + A[i, k] * B[k, j]

            out[i, j] = s

    return out

def test_dot():
    A = np.random.random((10, 10))
    B = np.random.random((10, 10))
    C = np.empty_like(A)

    assert np.allclose(np.dot(A, B), ndot(A, B, C))

if __name__ == '__main__':
    test_dot()
