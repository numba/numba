from __future__ import print_function, absolute_import, division

import numpy as np

from numba import config, cuda, jit
from numba.cuda.testing import unittest, SerialMixin, skip_on_cudasim


def foo(ary):
    x = cuda.threadIdx.x
    if x == 1:
        # NOTE: indexing with a out-of-bounds constant can fail at
        # compile-time instead (because the getitem is rewritten as a static_getitem)
        # XXX: -1 is actually a valid index for a non-empty tuple...
        ary.shape[-x]


class TestException(SerialMixin, unittest.TestCase):
    def test_exception(self):
        unsafe_foo = cuda.jit(foo)
        safe_foo = cuda.jit(debug=True)(foo)

        if not config.ENABLE_CUDASIM:
            # Simulator throws exceptions regardless of debug
            # setting
            unsafe_foo[1, 2](np.array([0, 1]))

        with self.assertRaises(IndexError) as cm:
            safe_foo[1, 2](np.array([0, 1]))
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

    @skip_on_cudasim("failing case doesn't happen in CUDASIM")
    @unittest.expectedFailure
    def test_raise_causing_warp_diverge_failing(self):
        """Test case for issue #2655.

        This test that the issue still exists in debug mode.
        """
        self.case_raise_causing_warp_diverge(with_debug_mode=True)


if __name__ == '__main__':
    unittest.main()
