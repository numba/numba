# adapted from cython/tests/run/builtin_abs.pyx

"""
>>> _abs = abs_as_name()
>>> _abs(-5)
5

>>> py_abs(-5)
5
>>> py_abs(-5.5)
5.5

>>> int_abs(-5)
5
>>> long_abs(-5)
5

>>> long_long_abs(-(2**33)) == 2**33
True

>>> double_abs(-5)
5.0
>>> double_abs(-5.5)
5.5

>>> float_abs(-5)
5.0
>>> float_abs(-5.5)
5.5
"""

from numba import *

@jit(backend='ast')
def abs_as_name():
    x = abs
    return x

@jit(backend='ast', argtypes=[object_])
def py_abs(a):
    return abs(a)

@jit(backend='ast', argtypes=[int_])
def int_abs(a):
    return abs(a)

@jit(backend='ast', argtypes=[long_])
def long_abs(a):
    return abs(a)

@jit(backend='ast', argtypes=[longlong])
def long_long_abs(a):
    return abs(a)

@jit(backend='ast', argtypes=[double])
def double_abs(a):
    return abs(a)

@jit(backend='ast', argtypes=[float_])
def float_abs(a):
    return abs(a)

if __name__ == '__main__':
    import doctest
    doctest.testmod()