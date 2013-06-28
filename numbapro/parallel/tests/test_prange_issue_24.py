import numpy as np
from numbapro import autojit, jit, prange
from numba import double, void, uint32
from .support import testcase, main

@testcase
def test_prange_redux():
    @autojit
    def prange_redux():
        c = 0
        a = 1
        for i in prange(10):
            c += a
        return c

if __name__ == '__main__':
    main()
