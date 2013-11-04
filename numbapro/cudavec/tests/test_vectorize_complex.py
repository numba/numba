import numpy as np
from numbapro import vectorize
from .support import testcase, main, assertTrue

@testcase
def test_vectorize_complex():
    @vectorize(['complex128(complex128)'], target='gpu')
    def vcomp(a):
        return a * a + 1.

    A = np.arange(5, dtype=np.complex128)
    B = vcomp(A)
    assertTrue(np.allclose(A * A + 1., B))

if __name__ == '__main__':
    main()
