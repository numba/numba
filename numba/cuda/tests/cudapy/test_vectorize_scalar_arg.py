from __future__ import absolute_import, print_function, division
import numpy as np
from numba import vectorize
from numba import cuda, float64
from numba import unittest_support as unittest
from numba.cuda.testing import skip_on_cudasim, SerialMixin
from numba import config

sig = [float64(float64, float64)]


target='cuda'
if config.ENABLE_CUDASIM:
    target='cpu'


@skip_on_cudasim('ufunc API unsupported in the simulator')
class TestCUDAVectorizeScalarArg(SerialMixin, unittest.TestCase):

    def test_vectorize_scalar_arg(self):
        @vectorize(sig, target=target)
        def vector_add(a, b):
            return a + b

        A = np.arange(10, dtype=np.float64)
        dA = cuda.to_device(A)
        v = vector_add(1.0, dA)

        np.testing.assert_array_almost_equal(
            v.copy_to_host(),
            np.arange(1, 11, dtype=np.float64))

    def test_vectorize_all_scalars(self):
        @vectorize(sig, target=target)
        def vector_add(a, b):
            return a + b

        v = vector_add(1.0, 1.0)

        np.testing.assert_almost_equal(2.0, v)


if __name__ == '__main__':
    unittest.main()
