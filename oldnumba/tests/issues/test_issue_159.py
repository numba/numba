# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import


from numba import *
import numpy as np

@jit(void(f8[:]))
def ff(T):
    for j in range(100): #reduce 100 to 10 get no error
        T[j]=1.0

x=np.ones(100,dtype=np.double)
ff(x)
assert np.all(x == 1.0)
