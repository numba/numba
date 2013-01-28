import numpy as np

import numba
from numba import *

@autojit
def numba_dot(A, B):
    result = np.dot(A, B)
    return numba.typeof(result), result

def test_numba_dot():
    A = np.array(1)
    B = np.array(2)

    for i in range(1, 10):
        for j in range(1, 10):
            shape_A = (1,) * i
            shape_B = (1,) * j

            x = A.reshape(*shape_A)
            y = B.reshape(*shape_B)

            result_type, result = numba_dot(x, y)

            assert result == np.dot(x, y)
            assert result.ndim == result_type.ndim
            # assert result.dtype == result_type.get_dtype()

if __name__ == "__main__":
    test_numba_dot()