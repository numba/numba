"""
>>> pow3(2,3,5)
3
>>> pow3(3,3,5)
2
 
>>> pow3_const()
3
 
>>> pow2(2,3)
8
>>> pow2(3,3)
27
>>> pow2(3.0,3)
27.0
>>> pow2(3,3.0)
27.0
>>> pow2(3.0,3.0)
27.0
>>> pow2(1.5, 2)
2.25
>>> pow2(1.5, 1.5) == pow(1.5, 1.5)
True
 
>>> pow_op(3,3)
27
>>> pow_op(3.0,3)
27.0
>>> pow_op(3,3.0)
27.0
>>> pow_op(3.0,3.0)
27.0
>>> pow_op(1.5, 2)
2.25
>>> pow_op(1.5, 1.5) == pow(1.5, 1.5)
True

>>> pow2_const()
8

>>> c1, c2 = 1.2 + 4.1j, 0.6 + 0.5j
>>> allclose(pow2(c1, c2), pow(c1, c2))
True

>>> d1, d2 = 4.2, 5.1
>>> allclose(pow2(d1, d2), pow(d1, d2))
True
"""

from numba import autojit
from numpy import allclose
 
@autojit
def pow3(a,b,c):
    return pow(a,b,c)
 
@autojit
def pow3_const():
    return pow(2,3,5)
 
@autojit(nopython=True)
def pow2(a,b):
    return pow(a,b)
 
@autojit(nopython=True)
def pow_op(a,b):
    return a**b
 
@autojit(nopython=True)
def pow2_const():
    return pow(2,3)
 
if __name__ == '__main__':
    import numba
    numba.testing.testmod()