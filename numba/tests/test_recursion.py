from numba import *
import numba as nb

@jit(int_(int_))
def fac(arg):
    if arg == 1:
        return 1
    else:
        return arg * fac(arg - 1)

assert fac(10) == fac.py_func(10)