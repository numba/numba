"""
>>> empty_str()
''
>>> str_convert(12.2)
'12.2'
"""

import sys

from numba import *

@autojit(backend='ast')
def empty_str():
    x = str()
    return x

@autojit(backend='ast')
def str_convert(x):
    return str(x)

if __name__ == '__main__':
    import numba
    numba.testing.testmod()
