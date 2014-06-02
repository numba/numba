from __future__ import absolute_import, print_function, division
import numpy as np
from numbapro import vectorize
from numbapro import cuda, float64
from numbapro.testsupport import unittest

sig = [float64(float64, float64)]


@vectorize(sig, target='gpu')
def vector_add(a, b):
    return a + b


class TestCUDAVectorizeScalarArg(unittest.TestCase):
    def test_vectorize_scalar_arg(self):
        A = np.arange(10, dtype=np.float64)
        dA = cuda.to_device(A)
        vector_add(1.0, dA)

    def test_vectorize_all_scalars(self):
        vector_add(1.0, 1.0)


if __name__ == '__main__':
    unittest.main()
