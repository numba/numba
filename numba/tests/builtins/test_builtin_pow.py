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

>>> pow2_const()
8

>>> c1, c2 = 1.2 + 4.1j, 0.6 + 0.5j
>>> pow2(c1, c2) == pow(c1, c2)
True

>>> d1, d2 = 4.2, 5.1
>>> pow2(d1, d2) == pow(d1, d2)
True
"""

from numba import *

@autojit(backend='ast')
def pow3(a,b,c):
    return pow(a,b,c)

@autojit(backend='ast')
def pow3_const():
    return pow(2,3,5)

@autojit(backend='ast')
def pow2(a,b):
    return pow(a,b)

@autojit(backend='ast')
def pow2_const():
    return pow(2,3)

if __name__ == '__main__':
#    c1, c2 = 1.2 + 4.1j, 0.6 + 0.5j
#    print pow2(c1, c2)
    import doctest
    doctest.testmod()