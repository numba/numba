# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from numba import jit, autojit

@jit('f8(f8,f8)')
def addv(a,b): return a+b

@jit('f8(f8[:])')
def sum1d(A):
    n = A.size
    s = 0.0
    for i in range(n):
        a = A[i]
        s = addv(s, A[i])
    return s

