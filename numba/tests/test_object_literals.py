"""
>>> get_list('world')
[1, 'hello', 2.0, 'world']
>>> get_tuple('world')
(1, 'hello', 2.0, 'world')
>>> get_dict('world') == {"hello": 1, 2.0: 'world'}
True
"""

import sys
from numba import *

myglobal = 20

@autojit(backend='ast')
def get_list(x):
    return [1, "hello", 2.0, x]

@autojit(backend='ast')
def get_tuple(x):
    return (1, "hello", 2.0, x)

@autojit(backend='ast')
def get_dict(x):
    return {"hello": 1, 2.0: x}

if __name__ == '__main__':
    import numba
    from numba.testing.test_support import rewrite_doc
    __doc__ = rewrite_doc(__doc__)
    numba.testing.testmod()
