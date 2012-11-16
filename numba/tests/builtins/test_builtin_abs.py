# adapted from cython/tests/run/builtin_abs.pyx

"""
>>> _abs = abs_as_name()
>>> _abs(-5)
5

>>> py_abs(-5)
5
>>> py_abs(-5.5)
5.5

>>> long(int32_abs(-5))
10L
>>> long(int_abs(-5))
10L
>>> long(long_abs(-5))
10L
>>> long(ulong_abs(5))
10L

>>> long_long_abs(-(2**33)) == 2**34
True
>>> ulong_long_abs(2**33) == 2**34
True

>>> double_abs(-5)
10.0
>>> double_abs(-5.5)
11.0

>>> float_abs(-5)
10.0
>>> float_abs(-5.5)
11.0

>>> round(complex64_abs(-10-2j), 2)
20.4
>>> round(complex128_abs(-10-2j), 2)
20.4
"""

from numba import *

@jit(backend='ast')
def abs_as_name():
    x = abs
    return x

@autojit(backend='ast')
def _abs(value):
    result = abs(value)
    with nopython:
        return result * 2 # test return type being non-object

@jit(backend='ast', argtypes=[object_])
def py_abs(a):
    return abs(a)

@jit(backend='ast', argtypes=[int_])
def int_abs(a):
    return _abs(a)

@jit(backend='ast', argtypes=[long_])
def long_abs(a):
    return _abs(a)

@jit(backend='ast', argtypes=[ulong])
def ulong_abs(a):
    return _abs(a)

@jit(backend='ast', argtypes=[int32])
def int32_abs(a):
    return _abs(a)

@jit(backend='ast', argtypes=[longlong])
def long_long_abs(a):
    return _abs(a)

@jit(backend='ast', argtypes=[ulonglong])
def ulong_long_abs(a):
    return _abs(a)

@jit(backend='ast', argtypes=[double])
def double_abs(a):
    return _abs(a)

@jit(backend='ast', argtypes=[float_])
def float_abs(a):
    return _abs(a)

@jit(backend='ast', argtypes=[complex64])
def complex64_abs(a):
    return _abs(a)


@jit(backend='ast', argtypes=[complex128])
def complex128_abs(a):
    return _abs(a)

if __name__ == '__main__':
    # print long_long_abs(-(2**33)), 2**33
    import doctest
    doctest.testmod()
