import numpy as np

from numba import njit


@njit(cache=True)
def f3(x):
    return np.ascontiguousarray(x.transpose(2, 0, 1))
    return fn

@njit(cache=True)
def f4(x):
    return x.reshape((1,) + x.shape)


@njit
def run(x):
    return f3(x), f4(x)

r3, r4 = run(np.zeros((2, 3, 4)))
print('r3', r3)
print('r4', r4)


