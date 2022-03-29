import numpy as np

from numba.cuda.testing import CUDATestCase
from numba import cuda

from numba.cuda import make_float3


class TestCudaVectorType(CUDATestCase):
    def test_creation_readout(self):

        @cuda.jit
        def kernel(res):
            f3 = make_float3(1.0, 2.0, 3.0)
            res[0] = f3.x
            res[1] = f3.y
            res[2] = f3.z
        kernel[1, 1]()

        arr = np.zeros((3,))
        kernel[1, 1](arr)
        self.assertAlmostEquals(arr, np.array([1.0, 2.0, 3.0]))

    # def test_write_attr(self):
    #     @cuda.jit
    #     def kernel():
    #         f3 = make_float3(1.0, 2.0, 3.0)
    #         f3.x = 42.0
    #     kernel[1, 1]()

    # def test_copy_ctor():

    #     @cuda.jit
    #     def kernel():
    #         f3 = make_float3(1.0, 2.0, 3.0)
    #         f3_2 = float3(f3)
