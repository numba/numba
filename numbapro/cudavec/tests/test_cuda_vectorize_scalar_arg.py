import numpy as np
from numbapro import vectorize
from numbapro import cuda, float64
from .support import testcase, main

sig = [float64(float64, float64)]

@vectorize(sig, target='gpu')
def vector_add(a, b):
    return a + b

@testcase
def test_vectorize_scalar_arg():
    A = np.arange(10, dtype=np.float64)
    dA = cuda.to_device(A)
    vector_add(1.0, dA)

@testcase
def test_vectorize_all_scalars():
    vector_add(1.0, 1.0)

if __name__ == '__main__':
    main()
