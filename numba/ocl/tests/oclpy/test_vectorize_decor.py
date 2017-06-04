from __future__ import absolute_import, print_function, division

import numpy as np

from numba import unittest_support as unittest
from numba import vectorize, ocl
from numba.tests.npyufunc import test_vectorize_decor
from numba.ocl.testing import skip_on_oclsim


class TestVectorizeDecor(test_vectorize_decor.BaseVectorizeDecor):
    def test_gpu_1(self):
        self._test_template_1('ocl')

    def test_gpu_2(self):
        self._test_template_2('ocl')

    def test_gpu_3(self):
        self._test_template_3('ocl')


class TestGPUVectorizeBroadcast(unittest.TestCase):
    def test_broadcast_bug_90(self):
        """
        https://github.com/ContinuumIO/numbapro/issues/90
        """

        a = np.random.randn(100, 3, 1)
        b = a.transpose(2, 1, 0)

        def fn(a, b):
            return a - b

        @vectorize(['float64(float64,float64)'], target='ocl')
        def fngpu(a, b):
            return a - b

        expect = fn(a, b)
        got = fngpu(a, b)
        np.testing.assert_almost_equal(expect, got)

    def test_device_broadcast(self):
        """
        Same test as .test_broadcast_bug_90() but with device array as inputs
        """

        a = np.random.randn(100, 3, 1)
        b = a.transpose(2, 1, 0)

        def fn(a, b):
            return a - b

        @vectorize(['float64(float64,float64)'], target='ocl')
        def fngpu(a, b):
            return a - b

        expect = fn(a, b)
        got = fngpu(ocl.to_device(a), ocl.to_device(b))
        np.testing.assert_almost_equal(expect, got.copy_to_host())


if __name__ == '__main__':
    unittest.main()
