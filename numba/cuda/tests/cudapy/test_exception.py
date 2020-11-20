import numpy as np

from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.core import config


class TestException(CUDATestCase):
    def test_exception(self):
        def foo(ary):
            x = cuda.threadIdx.x
            if x == 2:
                # NOTE: indexing with a out-of-bounds constant can fail at
                # compile-time instead (because the getitem is rewritten as a
                # static_getitem)
                ary.shape[-x]

        unsafe_foo = cuda.jit(foo)
        safe_foo = cuda.jit(debug=True)(foo)

        if not config.ENABLE_CUDASIM:
            # Simulator throws exceptions regardless of debug
            # setting
            unsafe_foo[1, 3](np.array([0, 1]))

        with self.assertRaises(IndexError) as cm:
            safe_foo[1, 3](np.array([0, 1]))
        self.assertIn("tuple index out of range", str(cm.exception))

    def test_user_raise(self):
        @cuda.jit(debug=True)
        def foo(do_raise):
            if do_raise:
                raise ValueError

        foo[1, 1](False)
        with self.assertRaises(ValueError):
            foo[1, 1](True)

    def case_raise_causing_warp_diverge(self, with_debug_mode):
        """Testing issue #2655.

        Exception raising code can cause the compiler to miss location
        of unifying branch target and resulting in unexpected warp
        divergence.
        """
        @cuda.jit(debug=with_debug_mode)
        def problematic(x, y):
            tid = cuda.threadIdx.x
            ntid = cuda.blockDim.x

            if tid > 12:
                for i in range(ntid):
                    y[i] += x[i] // y[i]

            cuda.syncthreads()
            if tid < 17:
                for i in range(ntid):
                    x[i] += x[i] // y[i]

        @cuda.jit
        def oracle(x, y):
            tid = cuda.threadIdx.x
            ntid = cuda.blockDim.x

            if tid > 12:
                for i in range(ntid):
                    if y[i] != 0:
                        y[i] += x[i] // y[i]

            cuda.syncthreads()
            if tid < 17:
                for i in range(ntid):
                    if y[i] != 0:
                        x[i] += x[i] // y[i]

        n = 32
        got_x = 1. / (np.arange(n) + 0.01)
        got_y = 1. / (np.arange(n) + 0.01)
        problematic[1, n](got_x, got_y)

        expect_x = 1. / (np.arange(n) + 0.01)
        expect_y = 1. / (np.arange(n) + 0.01)
        oracle[1, n](expect_x, expect_y)

        np.testing.assert_almost_equal(expect_x, got_x)
        np.testing.assert_almost_equal(expect_y, got_y)

    def test_raise_causing_warp_diverge(self):
        """Test case for issue #2655.
        """
        self.case_raise_causing_warp_diverge(with_debug_mode=False)


if __name__ == '__main__':
    unittest.main()
