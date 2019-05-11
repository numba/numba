import numpy as np
import numba as nb

@nb.njit
def eattuple(a, b):
    return len(a), b+1

a = tuple(range(10))
b = 30
print(eattuple(a, b))
