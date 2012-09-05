from numba.decorators import jit
from numba import *

@jit(arg_types=[d[:], d[:]])
def copy_d(src, dst):
    N = src.shape[0]
    for i in range(N):
        dst[i] = src[i]


