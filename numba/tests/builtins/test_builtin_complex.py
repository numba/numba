"""
>>> empty_complex()
0j
>>> new_complex(1., 5)
(1+5j)
>>> convert_to_complex(10)
(10+0j)
>>> convert_to_complex(10+2j)
(10+2j)
>>> convert_to_complex(10.0)
(10+0j)
"""

import sys

from numba import *

@autojit(backend='ast')
def empty_complex():
    x = complex()
    return x

@autojit(backend='ast')
def new_complex(x, y):
    return complex(x, y)

@autojit(backend='ast')
def convert_to_complex(x):
    return complex(x)

if __name__ == '__main__':
    import numba
    numba.testing.testmod()
