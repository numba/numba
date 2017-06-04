from __future__ import print_function, absolute_import
import math
from numba import ocl
from numba.ocl.testing import unittest


class TestOclMonteCarlo(unittest.TestCase):
    def test_montecarlo(self):
        """Just make sure we can compile this
        """

        @ocl.jit(
            'void(double[:], double[:], double, double, double, double[:])')
        def step(last, paths, dt, c0, c1, normdist):
            i = ocl.grid(1)
            if i >= paths.shape[0]:
                return
            noise = normdist[i]
            paths[i] = last[i] * math.exp(c0 * dt + c1 * noise)


if __name__ == '__main__':
    unittest.main()

