from __future__ import print_function

import numpy as np

from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim


@cuda.jit
def this_grid(A):
    cuda.this_grid()
    A[0] = 1.0


@cuda.jit
def sync_group(A):
    g = cuda.this_grid()
    g.sync()
    A[0] = 1.0


@cuda.jit
def no_sync(A):
    A[0] = cuda.grid(1)


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


if __name__ == '__main__':
    unittest.main()
