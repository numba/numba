# -*- coding: utf-8 -*-
from numba import void, double, jit
import numpy as np
 
# thanks to @ufechner7

def multiassign(res0, res1, val0, val1):
    res0[:], res1[:] = val0[:], val1[:]

if __name__ == "__main__":
    multiassign1 = jit(void(double[:], double[:], double[:], double[:]))(multiassign)  
    res0  = np.zeros(2)
    res1  = np.zeros(2)
    val0 = np.array([0.0,0.0])
    val1 = np.array([1.0,1.0])
    multiassign1(res0, res1, val0, val1)
    assert (res0 == val0).all()
    assert (res1 == val1).all()
