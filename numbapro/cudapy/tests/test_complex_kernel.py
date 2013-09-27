import numpy as np
from numbapro import cuda

from .support import testcase, main, assertTrue

@testcase
def test_cuda_complex_arg():

    @cuda.jit('void(complex128[:], complex128)')
    def foo(a, b):
        i = cuda.grid(1)
        a[i] += b


    a = np.arange(5, dtype=np.complex128)
    a0 = a.copy()
    foo[1, a.shape](a, 2j)
    assertTrue(np.allclose(a, a0 + 2j))

if __name__ == '__main__':
    main()


