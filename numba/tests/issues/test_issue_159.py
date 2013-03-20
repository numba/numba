from numba import *
import numpy as np

@jit(void(f8[:]))
def ff(T):
    for j in range(100): #reduce 100 to 10 get no error
        T[j]=1.0

x=np.ones(100,dtype=np.double)
ff(x)