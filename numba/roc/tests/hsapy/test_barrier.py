import numpy as np

from numba import roc, float32
import unittest


class TestBarrier(unittest.TestCase):
    def test_proper_lowering(self):
        @roc.jit("void(float32[::1])")
        def twice(A):
            i = roc.get_global_id(0)
            d = A[i]
            roc.barrier(roc.CLK_LOCAL_MEM_FENCE)  # local mem fence
            A[i] = d * 2

        N = 256
        arr = np.random.random(N).astype(np.float32)
        orig = arr.copy()

        twice[2, 128](arr)

        # Assembly contains barrier instruction?
        self.assertIn("s_barrier", twice.assembly)
        # The computation is correct?
        np.testing.assert_allclose(orig * 2, arr)

    def test_no_arg_barrier_support(self):
        @roc.jit("void(float32[::1])")
        def twice(A):
            i = roc.get_global_id(0)
            d = A[i]
            # no argument defaults to global mem fence
            # which is the same for local in hsail
            roc.barrier()
            A[i] = d * 2

        N = 256
        arr = np.random.random(N).astype(np.float32)
        orig = arr.copy()

        twice[2, 128](arr)

        # Assembly contains barrier instruction?
        self.assertIn("s_barrier", twice.assembly)
        # The computation is correct?
        np.testing.assert_allclose(orig * 2, arr)

    def test_local_memory(self):
        blocksize = 10

        @roc.jit("void(float32[::1])")
        def reverse_array(A):
            sm = roc.shared.array(shape=blocksize, dtype=float32)
            i = roc.get_global_id(0)

            # preload
            sm[i] = A[i]
            # barrier
            roc.barrier(roc.CLK_LOCAL_MEM_FENCE)  # local mem fence
            # write
            A[i] += sm[blocksize - 1 - i]

        arr = np.arange(blocksize).astype(np.float32)
        orig = arr.copy()

        reverse_array[1, blocksize](arr)

        expected = orig[::-1] + orig
        np.testing.assert_allclose(expected, arr)


if __name__ == '__main__':
    unittest.main()
