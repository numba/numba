# adapted from cython/tests/run/builtin_abs.pyx

"""
>>> _abs = abs_as_name()
>>> _abs(-5)
5

>>> py_abs(-5)
5
>>> py_abs(-5.5)
5.5

>>> int(int32_abs(-5))
10
>>> int(int_abs(-5))
10
>>> int(long_abs(-5))
10
>>> int(ulong_abs(5))
10

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

>>> '%.2f' % round(complex64_abs(-10-2j), 2)
'20.40'
>>> '%.2f' % round(complex128_abs(-10-2j), 2)
'20.40'
"""

from numba import *

### Python usage

@jit(object_())
def abs_as_name():
    x = abs
    return x

@jit(argtypes=[object_])
def py_abs(a):
    return abs(a)

### nopython usage

@autojit(nopython=True)
def _abs(value):
    result = abs(value)
    with nopython:
        return result * 2 # test return type being non-object

@jit(nopython=True, argtypes=[int_])
def int_abs(a):
    return _abs(a)

@jit(nopython=True, argtypes=[long_])
def long_abs(a):
    return _abs(a)

@jit(nopython=True, argtypes=[ulong])
def ulong_abs(a):
    return _abs(a)

@jit(nopython=True, argtypes=[int32])
def int32_abs(a):
    return _abs(a)

@jit(nopython=True, argtypes=[longlong])
def long_long_abs(a):
    return _abs(a)

@jit(nopython=True, argtypes=[ulonglong])
def ulong_long_abs(a):
    return _abs(a)

@jit(nopython=True, argtypes=[double])
def double_abs(a):
    return _abs(a)

@jit(nopython=True, argtypes=[float_])
def float_abs(a):
    return _abs(a)

@jit(nopython=True, argtypes=[complex64])
def complex64_abs(a):
    return _abs(a)


@jit(nopython=True, argtypes=[complex128])
def complex128_abs(a):
    return _abs(a)

if __name__ == '__main__':
#    print long(int32_abs(-5))
    import numba
    numba.testing.testmod()
