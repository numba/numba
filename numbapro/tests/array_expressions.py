import numbapro
from numba.decorators import function

import numpy as np

@function
def array_expr(a, b, c):
    return a + b * c

def test_array_expressions():
    a = np.arange(100).reshape(10, 10).astype(np.float32)
    assert np.all(array_expr(a, a, a) == a + a * a)

if __name__ == '__main__':
    test_array_expressions()