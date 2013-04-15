# -*- coding: utf-8 -*-
# from __future__ import division, absolute_import

# Thanks to Neal Becker

import numpy as np
from numba import *
from numba.vectorize import vectorize
from math import exp, log1p


@vectorize([f8(f8,f8)])
def log_exp_sum2 (a, b):
    if a >= b:
        return a + (exp (-(a-b)))
    else:
        return b + (exp (-(b-a)))
    ## return max (a, b) + log1p (exp (-abs (a - b)))


#@autojit
@jit(f8[:,:] (f8[:,:]))
def log_exp_sum (u):
    s = u.shape[1] # Test wraparound when implemented!
    if s == 1:
        return u[...,0]
    elif s == 2:
        return log_exp_sum2 (u[...,0], u[...,1])
    else:
        return log_exp_sum2 (
            log_exp_sum (u[...,:s/2]),
            log_exp_sum (u[...,s/2:]))


from timeit import timeit
L = 1000
N = 100
u = np.tile (np.log (np.ones (L)/L), (N, 1))
#v = log_exp_sum (u)
from timeit import timeit
print(timeit(
    'log_exp_sum(u)', 'from __main__ import u, log_exp_sum', number=50))
