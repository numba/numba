# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from numba import jit

@jit('i4(i4,f8,i4,f8[:])')
def FindMult(i,u,p,U):
    s = 0
    for j in range(0-p, p+2):
        if U[i+j] == u:
            s = s + 1
    return s
