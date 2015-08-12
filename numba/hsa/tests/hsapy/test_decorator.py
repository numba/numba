from __future__ import print_function, absolute_import

import numpy as np

import numba.unittest_support as unittest
from numba import hsa


class TestDecorators(unittest.TestCase):
    def test_kernel_jit(self):
        @hsa.jit("(float32[:], float32[:])")
        def copy_vector(dst, src):
            tid = hsa.get_global_id(0)
            if tid < dst.size:
                dst[tid] = src[tid]

        src = np.arange(10, dtype=np.uint32)
        dst = np.zeros_like(src)
        copy_vector[10, 1](dst, src)
        np.testing.assert_equal(dst, src)

    def test_device_jit(self):
        @hsa.jit("float32(float32[:], intp)", device=True)
        def inner(src, idx):
            return src[idx]

        @hsa.jit("(float32[:], float32[:])")
        def outer(dst, src):
            tid = hsa.get_global_id(0)
            if tid < dst.size:
                dst[tid] = inner(src, tid)

        src = np.arange(10, dtype=np.uint32)
        dst = np.zeros_like(src)
        outer[10, 1](dst, src)
        np.testing.assert_equal(dst, src)

    def test_autojit_kernel(self):
        @hsa.jit
        def copy_vector(dst, src):
            tid = hsa.get_global_id(0)
            if tid < dst.size:
                dst[tid] = src[tid]

        for dtype in [np.uint32, np.float32]:
            src = np.arange(10, dtype=dtype)
            dst = np.zeros_like(src)
            copy_vector[10, 1](dst, src)
            np.testing.assert_equal(dst, src)


if __name__ == '__main__':
    unittest.main()
