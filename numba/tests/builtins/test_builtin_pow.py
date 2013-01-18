"""
>>> pow3(2,3,5) == 3
True
>>> pow3(3,3,5) == 2
True

>>> pow3_const() == 3
True

>>> pow2(2,3) == 8
True
>>> pow2(3,3) == 27
True

>>> pow2_const() == 8
True

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
#    pow2(1.0, 2.0)
#    import logging; logging.getLogger().setLevel(0)
#    c1, c2 = 1.2 + 4.1j, 0.6 + 0.5j
#    pow2(c1, c2) == pow(c1, c2)
#    pow3(2,3,5)
    import doctest
    doctest.testmod()
