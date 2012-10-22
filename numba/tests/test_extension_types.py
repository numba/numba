"""
>>> obj = MyExtension(10.0)
>>> obj.value
10.0
>>> obj._numba_attrs.value
10.0
"""

import sys

from numba import extension_types
from numba import *

def test(struct_type, vtab_type):
    """
    >>> struct_type = struct([('x', int_), ('y', char), ('z', float_)])
    >>> vtab_type = struct([])
    >>> obj, attrs = test(struct_type, vtab_type)
    Test
    hello!
    >>> attrs.x, attrs.y, attrs.z
    (0, 0, 0.0)
    """

    def init(self, value):
        self.value = value

    def repr(self):
        return str(self.value)

    dict = {
        '__init__': init,
        '__repr__': repr,
    }

    t = extension_types.create_new_extension_type('Test', (object,), dict,
                                                  struct_type, vtab_type,
                                                  [], [])
    print t.__name__
    print t('hello!')

    obj = t(10)
    return obj, obj._numba_attrs

@autojit
class MyExtension(object):
    def __init__(self, myfloat):
        self.value = myfloat

if __name__ == '__main__':
    import doctest
    doctest.testmod()