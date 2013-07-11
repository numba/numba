import numpy as np
from numbapro import cuda, vectorize
from .support import testcase, main

@testcase
def test_vectorize_complex():
    @vectorize(['complex128(complex128)'], target='gpu')
    def vcomp(a):
        return a * a + 1.

    A = np.arange(5, dtype=np.complex128)
    B = vcomp(A)
    assert np.allclose(A * A + 1., B)

if __name__ == '__main__':
    main()
