from __future__ import division, print_function, absolute_import

from numba import cfunc, jit


Z = 1

add_sig = "float64(float64, float64)"

div_sig = "float64(int64, int64)"


@cfunc(add_sig, cache=True, nopython=True)
def add_usecase(x, y):
    return x + y + Z

@cfunc(add_sig, nopython=True)
def add_nocache_usecase(x, y):
    return x + y + Z

@cfunc(div_sig, cache=True, nopython=True)
def div_usecase(a, b):
    return a / b


@jit(nopython=True)
def inner(x, y):
    return x + y + Z

@cfunc(add_sig, cache=True, nopython=True)
def outer(x, y):
    return inner(-y, x)
