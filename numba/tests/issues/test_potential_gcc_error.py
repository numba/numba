# This tests a potential GCC 4.1.2 miscompile of LLVM.
# The problem is observed as a error in greedy register allocation pass,
# which resulted as a segfault.
# No such problem in GCC 4.4.6.


from numba import *
import numpy as np

@jit(uint8[:,:](f8, f8, f8, f8, uint8[:,:], int32))
def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    return image



