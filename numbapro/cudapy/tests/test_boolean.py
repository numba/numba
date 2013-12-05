import numpy as np
from .support import testcase, main, assertTrue
from numbapro import cuda


def boolean_test(A, vertial):
    if vertial:
        A[0] = 123
    else:
        A[0] = 321

@testcase
def test_boolean():
    func = cuda.jit('void(float64[:], bool_)')(boolean_test)
    A = np.array([0], dtype='float64')
    func(A, True)
    assertTrue(A[0] == 123)
    func(A, False)
    assertTrue(A[0] == 321)

if __name__ == '__main__':
    main()
