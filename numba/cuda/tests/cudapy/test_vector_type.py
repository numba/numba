import numpy as np

from numba.cuda.testing import CUDATestCase

from numba import cuda, int8
from numba.cuda import float3, uint4, char3
from numba.cuda.vector_types import vector_types

def make_kernel(vtype):
    vobj = vtype.user_facing_object
    base_type = vtype.base_type
    def kernel_1elem(res):
        v = vobj(base_type(0))
        res[0] = v.x

    def kernel_2elem(res):
        v = vobj(base_type(0), base_type(1))
        res[0] = v.x
        res[1] = v.y

    def kernel_3elem(res):
        v = vobj(base_type(0), base_type(1), base_type(2))
        res[0] = v.x
        res[1] = v.y
        res[2] = v.z
    
    def kernel_4elem(res):
        v = vobj(
            base_type(0),
            base_type(1),
            base_type(2),
            base_type(3)
        )
        res[0] = v.x
        res[1] = v.y
        res[2] = v.z
        res[3] = v.w
    
    host_function = {1: kernel_1elem,
                    2: kernel_2elem,
                    3: kernel_3elem,
                    4: kernel_4elem
                }[vtype.num_elements]
    return cuda.jit(host_function)

class TestCudaVectorType(CUDATestCase):

    def setUp(self):
        """Compile a empty kernel to initialize vector types."""
        super().setUp()
        @cuda.jit("()")
        def k():
            pass

    def test_creation_readout(self):
        for vty in vector_types:
            with self.subTest(vty=vty):
                arr = np.zeros((vty.num_elements,))
                kernel = make_kernel(vty)
                kernel[1, 1](arr)
                np.testing.assert_almost_equal(
                    arr, np.array(range(vty.num_elements))
                )