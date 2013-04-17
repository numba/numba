import numpy as np
from numbapro import autojit, jit, prange


@autojit
def test_prange_modulo(a, b):
    c = np.zeros(10)
    for i in prange(c.size):
        c[i] = a % b
    return c

if __name__ == '__main__':
    assert np.all(test_prange_modulo(100, 48) == 4)

