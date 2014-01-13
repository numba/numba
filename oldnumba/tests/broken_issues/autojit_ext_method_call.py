# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

# Thanks to Neal Becker

# TODO: Fix and dedup

import numpy as np
from numba import *

f8_array_ty = f8[:]

@autojit
class fir (object):
    ## @void(f8[:])
    def __init__ (self, coef):
        self.coef = coef
        self.inp = np.zeros_like (coef)

    ## @void(f8)
    def shift (self, u):
        size = self.inp.size
        for i in range (size-1-1, -1, -1):
            self.inp[i+1] = self.inp[i]
        self.inp[0] = u

    ## @double ()
    def compute1(self):
        s = 0
        size = self.coef.size
        for i in range (size):
            s += self.inp[i] * self.coef[i]
        return s

    ## @f8(f8)
    def shift_compute1 (self, u):
        self.shift (u)
        return self.compute1()

    ## @f8_array_ty(f8_array_ty)
    def compute (self, u):
        out = np.empty_like (u)
        size = u.size
        for i in range (size):
            out[i] = self.shift_compute1 (u[i])
        return out

    ## @f8_array_ty(f8_array_ty)
    def __call__ (self, u):
        return self.compute (u)

if __name__ == '__main__':
    from timeit import timeit

    coef = np.arange(100, dtype=np.double)
    filt = fir (coef)

    inp = np.arange(100, dtype=np.double)
    out = filt.compute (inp)

    filt (coef)

