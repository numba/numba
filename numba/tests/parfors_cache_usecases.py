import numpy as np
from numba import njit
from numba.npyufunc.parallel import _launch_threads

_launch_threads()  # FIXME


@njit(parallel=True, cache=True)
def arrayexprs(arr):
    return arr / arr.sum()
