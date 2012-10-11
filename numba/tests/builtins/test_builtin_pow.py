# adapted from cython/tests/run/builtin_pow.pyx

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
"""

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
    import doctest
    doctest.testmod()