# Issue: #144

# Thanks to Neal Becker

import numpy as np
from numba import *

f8_array_ty = f8[:]

@jit
class fir (object):
    @void(f8[:])
    def __init__ (self, coef):
        self.coef = coef
        self.inp = np.zeros_like (coef)

    @void(f8_array_ty)
    def shift (self, u):
        size = self.inp.size
        n = len (u)
        for i in range (size-1-n, -1, -1):
            self.inp[i+n] = self.inp[i]
        self.inp[:n] = u

    @f8 ()
    def compute1(self):
        s = 0
        size = self.coef.size
        for i in range (size):
            s += self.inp[i] * self.coef[i]
        return s

    @f8(f8_array_ty)
    def shift_compute1 (self, u):
        self.shift (u)
        return self.compute1()

    @f8_array_ty(f8_array_ty)
    def compute (self, u):
        out = np.empty_like (u)
        size = u.size
        for i in range (size):
            out[i] = self.shift_compute1 (u[i:i+1])
        return out

    @f8_array_ty(f8_array_ty)
    def __call__ (self, u):
        return self.compute (u)

if __name__ == '__main__':
    coef = np.random.rand (16)
    filt = fir (coef)
    inp = np.random.rand (1000)

    x = inp[:10].copy()
    out = filt.compute (inp)
    assert np.all(x == inp[:10])
