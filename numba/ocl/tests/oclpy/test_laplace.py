from __future__ import print_function, absolute_import, division
import numpy as np
import time
from numba import ocl, config, float64, void
from numba.ocl.testing import unittest

# NOTE: OpenCL kernel does not return any value

tpb = 16
SM_SIZE = tpb, tpb

class TestOclLaplace(unittest.TestCase):
    def test_laplace_small(self):

        @ocl.jit(void(float64[:, :], float64[:, :], float64[:, :]))
        def jocabi_relax_core(A, Anew, error):
            err_sm = ocl.shared.array(SM_SIZE, dtype=float64)

            ty = ocl.get_local_id(0)
            tx = ocl.get_local_id(1)
            bx = ocl.get_local_size(0)
            by = ocl.get_local_size(1)

            n = A.shape[0]
            m = A.shape[1]

            i = ocl.get_global_id(0)
            j = ocl.get_global_id(1)

            err_sm[ty, tx] = 0
            if j >= 1 and j < n - 1 and i >= 1 and i < m - 1:
                Anew[j, i] = 0.25 * ( A[j, i + 1] + A[j, i - 1] \
                                      + A[j - 1, i] + A[j + 1, i])
                err_sm[ty, tx] = Anew[j, i] - A[j, i]

            ocl.barrier()

            # max-reduce err_sm vertically
            t = tpb // 2
            while t > 0:
                if ty < t:
                    err_sm[ty, tx] = max(err_sm[ty, tx], err_sm[ty + t, tx])
                t //= 2
                ocl.barrier()

            # max-reduce err_sm horizontally
            t = tpb // 2
            while t > 0:
                if tx < t and ty == 0:
                    err_sm[ty, tx] = max(err_sm[ty, tx], err_sm[ty, tx + t])
                t //= 2
                ocl.barrier()

            if tx == 0 and ty == 0:
                error[by, bx] = err_sm[0, 0]



        NN, NM = 256, 256
        iter_max = 1000

        A = np.zeros((NN, NM), dtype=np.float64)
        Anew = np.zeros((NN, NM), dtype=np.float64)

        n = NN
        m = NM

        tol = 1.0e-6
        error = 1.0

        for j in range(n):
            A[j, 0] = 1.0
            Anew[j, 0] = 1.0

        timer = time.time()
        iter = 0

        blockdim = (tpb, tpb)
        griddim = (NN // blockdim[0], NM // blockdim[1])

        error_grid = np.zeros(griddim)

        stream = ocl.stream()

        dA = ocl.to_device(A, stream)          # to device and don't come back
        dAnew = ocl.to_device(Anew, stream)    # to device and don't come back
        derror_grid = ocl.to_device(error_grid, stream)

        while error > tol and iter < iter_max:
            self.assertTrue(error_grid.dtype == np.float64)

            jocabi_relax_core[griddim, blockdim, stream](dA, dAnew, derror_grid)

            derror_grid.copy_to_host(error_grid, stream=stream)


            # error_grid is available on host
            stream.finish()

            error = np.abs(error_grid).max()

            # swap dA and dAnew
            tmp = dA
            dA = dAnew
            dAnew = tmp

            iter += 1

        runtime = time.time() - timer


if __name__ == '__main__':
    unittest.main()
