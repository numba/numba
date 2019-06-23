from __future__ import absolute_import, print_function, division

import numpy as np

from numba import unittest_support as unittest
from numba.cuda.testing import SerialMixin
from numba import cuda, void, f8

from numba.cuda.graph import KernelNode, EmptyNode, MemcpyDtoHNode, MemcpyHtoDNode, MemsetNode


class TestCudaGraph(SerialMixin, unittest.TestCase):

    def test_cuda_kernel(self):
        arr = cuda.to_device(np.array([1.]))

        @cuda.jit(void(f8[:]))
        def k1(a):
            a[0] += 2

        @cuda.jit(void(f8[:]))
        def k2(a):
            a[0] *= 3

        n1 = KernelNode(k1, [arr])
        n2 = KernelNode(k2, [arr], [n1])
        n2.build().launch()
        cuda.synchronize()

        self.assertTrue(np.all(arr.copy_to_host() == [9]))

    def test_auto_cuda_kernel(self):
        arr = cuda.to_device(np.array([1.]))

        @cuda.jit
        def k1(a):
            a[0] += 2

        @cuda.jit
        def k2(a):
            a[0] *= 3

        n1 = KernelNode(k1, [arr])
        n2 = KernelNode(k2, [arr], [n1])
        n2.build().launch()
        cuda.synchronize()

        self.assertTrue(np.all(arr.copy_to_host() == [9]))

    def test_kernel_dim(self):
        arr = cuda.to_device(np.zeros(6))

        @cuda.jit
        def k1(a):
            a[0] = cuda.gridDim.x
            a[1] = cuda.gridDim.y
            a[2] = cuda.gridDim.z
            a[3] = cuda.blockDim.x
            a[4] = cuda.blockDim.y
            a[5] = cuda.blockDim.z

        n1 = KernelNode(k1, [arr])
        n1.build().launch()
        cuda.synchronize()

        self.assertTrue(np.all(arr.copy_to_host() == [1, 1, 1, 1, 1, 1]))

        n1 = KernelNode(k1, [arr], params={ 'gridDim': 4, 'blockDim': 5 })
        n1.build().launch()
        cuda.synchronize()

        self.assertTrue(np.all(arr.copy_to_host() == [4, 1, 1, 5, 1, 1]))

        n1 = KernelNode(k1, [arr], params={ 'gridDim': (1, 2, 3), 'blockDim': (4, 5, 6) })
        n1.build().launch()
        cuda.synchronize()

        self.assertTrue(np.all(arr.copy_to_host() == [1, 2, 3, 4, 5, 6]))

    def test_empty_node(self):
        arr = cuda.to_device(np.array([1.]))

        @cuda.jit
        def k1(a):
            a[0] += 2

        @cuda.jit
        def k2(a):
            a[0] *= 3

        n1 = KernelNode(k1, [arr])
        n2 = KernelNode(k2, [arr], [n1])
        n3 = EmptyNode([n2])
        n3.build().launch()
        cuda.synchronize()

        self.assertTrue(np.all(arr.copy_to_host() == [9]))

    def test_memcpy_node(self):
        harr = np.array([1.])
        darr = cuda.device_array_like(harr)

        @cuda.jit
        def k1(a):
            a[0] += 2

        n1 = MemcpyHtoDNode(darr, harr, harr.nbytes)
        n2 = KernelNode(k1, [darr], [n1])
        n3 = MemcpyDtoHNode(harr, darr, harr.nbytes, [n2])
        n3.build().launch()
        cuda.synchronize()

        self.assertTrue(np.all(harr == [3]))

    def test_memset_node(self):
        harr = np.array([0]).astype(np.int32)
        darr = cuda.device_array_like(harr)

        @cuda.jit
        def k1(a):
            a[0] += 2

        n1 = MemsetNode(darr, harr.nbytes, 1)
        n2 = KernelNode(k1, [darr], [n1])
        n3 = MemcpyDtoHNode(harr, darr, harr.nbytes, [n2])
        n3.build().launch()
        cuda.synchronize()

        self.assertTrue(np.all(harr == [3]))


if __name__ == '__main__':
    unittest.main()
