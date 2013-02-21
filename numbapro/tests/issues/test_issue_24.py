import numpy as np
import numbapro 
from numba import autojit, jit, double, void, uint32, prange

@autojit
def test_prange_redux():
    c = 0
    a = 1
    for i in prange(10):
        c += a
    return c

if __name__ == '__main__':
    print test_prange_redux()

