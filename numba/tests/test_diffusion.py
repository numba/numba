import unittest

import numpy as np
from numba import autojit 

mu = 0.1
Lx, Ly = 101, 101

@autojit 
def diffusionObstacleStep(u,tempU,iterNum):
    for n in range(iterNum):
        for i in range(1, Lx - 1):
            for j in range(1, Ly - 1):
                u[i,j] = mu * (tempU[i+1,j]-2*tempU[i,j]+tempU[i-1,j] +
                               tempU[i,j+1]-2*tempU[i,j]+tempU[i,j-1])

        # Bug in Meta??
        # tempU, u = u, tempU
        # -> Assign(targets=[Name(id='tempU', ctx=Store()),
        #                    Name(id='u', ctx=Store())],
        #           value=Name(id='u', ctx=Load()))
        temp = u
        u = tempU
        tempU = temp

def get_arrays():
    u = np.zeros([Lx, Ly], dtype=np.float64)
    tempU = np.zeros([Lx, Ly], dtype=np.float64)
    u[Lx / 2, Ly / 2] = 1000.0
    return tempU, u

def test_diffusion():
    tempU, u = get_arrays()
    iterNum = 10
    diffusionObstacleStep(u, tempU, iterNum)

    tempU_numpy, u_numpy = get_arrays()
    diffusionObstacleStep.py_func(u_numpy, tempU_numpy, iterNum)

    print u
    print u_numpy
    assert np.allclose(u, u_numpy)

if __name__ == "__main__":
    test_diffusion()
