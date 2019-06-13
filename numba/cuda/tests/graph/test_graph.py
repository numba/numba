from __future__ import absolute_import, print_function, division

import numpy as np

from numba import unittest_support as unittest
from numba.cuda.testing import SerialMixin
from numba import cuda, void, f8

from numba.cuda.graph import KernelNode

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

if __name__ == '__main__':
    unittest.main()
