from __future__ import absolute_import, print_function, division
import numpy as np
from numba import vectorize
from numba import ocl, float64
from numba import unittest_support as unittest
from numba.ocl.testing import skip_on_oclsim
from numba import config

sig = [float64(float64, float64)]


target='ocl'


class TestOCLVectorizeScalarArg(unittest.TestCase):

    def test_vectorize_scalar_arg(self):
        @vectorize(sig, target=target)
        def vector_add(a, b):
            return a + b

        A = np.arange(10, dtype=np.float64)
        dA = ocl.to_device(A)
        vector_add(1.0, dA)

    def test_vectorize_all_scalars(self):
        @vectorize(sig, target=target)
        def vector_add(a, b):
            return a + b
        
        vector_add(1.0, 1.0)


if __name__ == '__main__':
    unittest.main()
