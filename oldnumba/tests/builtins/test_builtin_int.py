"""
>>> empty_int()
0
>>> convert_int(2.5)
2
>>> convert_to_int('FF', 16)
255
"""

from numba import autojit

@autojit
def empty_int():
    return int()

@autojit
def convert_int(x):
    return int(x)

@autojit
def convert_to_int(s, base):
    return int(s, base)

if __name__ == '__main__':
    import numba
    numba.testing.testmod()