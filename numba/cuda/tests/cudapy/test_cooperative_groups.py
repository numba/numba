from __future__ import print_function

import numpy as np

from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim


@cuda.jit
def this_grid(A):
    cuda.cg.this_grid()
    A[0] = 1.0


@cuda.jit
def sync_group(A):
    g = cuda.cg.this_grid()
    g.sync()
    A[0] = 1.0


@cuda.jit
def no_sync(A):
    A[0] = cuda.grid(1)


@cuda.jit
def sequential_rows(M):
    # The grid writes rows one at a time. Each thread reads an element from
    # the previous row written by its "opposite" thread.
    #
    # A failure to sync the grid at each row would result in an incorrect
    # result as some threads could run ahead of threads in other blocks, or
    # fail to see the update to the previous row from their opposite thread.

    col = cuda.grid(1)
    g = cuda.cg.this_grid()

    rows = M.shape[0]
    cols = M.shape[1]

    for row in range(1, rows):
        opposite = cols - col - 1
        M[row, col] = M[row - 1, opposite] + 1
        g.sync()


@skip_on_cudasim("Cooperative groups not supported on simulator")
class TestCudaCooperativeGroups(CUDATestCase):
    def test_this_grid(self):
        A = np.full(1, fill_value=np.nan)
        this_grid[1, 1](A)

        # Ensure the kernel executed beyond the call to cuda.this_grid()
        self.assertFalse(np.isnan(A[0]), 'set it to something!')

        # this_grid should have been determinted to be cooperative
        for key, defn in this_grid.definitions.items():
            self.assertTrue(defn.cooperative)

    def test_sync_group(self):
        A = np.full(1, fill_value=np.nan)
        sync_group[1, 1](A)

        # Ensure the kernel executed beyond the call to cuda.sync_group()
        self.assertFalse(np.isnan(A[0]), 'set it to something!')

        # this_grid should have been determinted to be cooperative
        for key, defn in this_grid.definitions.items():
            self.assertTrue(defn.cooperative)

    def test_false_cooperative_doesnt_link_cudadevrt(self):
        """
        We should only mark a kernel as cooperative and link cudadevrt if the
        kernel uses grid sync. Here we ensure that one that doesn't use grid
        synsync isn't marked as such.
        """
        A = np.full(1, fill_value=np.nan)
        no_sync[1, 1](A)

        for key, defn in no_sync.definitions.items():
            self.assertFalse(defn.cooperative)
            for link in defn._func.linking:
                self.assertNotIn('cudadevrt', link)

    def test_sync_at_matrix_row(self):
        A = np.zeros((1024, 1024), dtype=np.int32)
        blockdim = 32
        griddim = A.shape[1] // blockdim

        sequential_rows[griddim, blockdim](A)

        reference = np.tile(np.arange(1024), (1024, 1)).T
        np.testing.assert_equal(A, reference)


if __name__ == '__main__':
    unittest.main()
