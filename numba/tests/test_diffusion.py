import unittest

import numpy as np
from numba import *

mu = 0.1
Lx, Ly = 101, 101

@autojit(backend='bytecode')
def diffusionObstacleStep(u,tempU,iterNum):
    for n in range(iterNum):
        a = Lx/2-3
        b = Lx/2-3
        for i in range(Lx-1):
            for j in range(Ly-1):
                if (i-a)**2+(j-b)**2 <  3:
                    tempU[i,j] = u[i,j]
                else:
                    tempU[i,j] = u[i,j] + mu * (u[i+1,j]-2*u[i,j]+u[i-1,j] +\
                                                u[i,j+1]-2*u[i,j]+u[i,j-1] )
        # Should support copying of arrays 
        for i in range(Lx-1):
            for j in range(Ly-1):
                u[i,j] = tempU[i,j]
                tempU[i,j] = 0.0

class TestDiffusion(unittest.TestCase):

    def get_arrays(self):
        u = np.zeros([Lx, Ly], dtype=np.float64)
        tempU = np.zeros([Lx, Ly], dtype=np.float64)
        u[Lx / 2, Ly / 2] = 1000.0
        return tempU, u

    def test_diffusion(self):
        tempU, u = self.get_arrays()
        iterNum = 10
        diffusionObstacleStep(u, tempU, iterNum)

        tempU_numpy, u_numpy = self.get_arrays()
        diffusionObstacleStep.py_func(u, tempU, iterNum)

        print u
        print u_numpy
        assert np.allclose(u, u_numpy)

if __name__ == "__main__":
    unittest.main()
