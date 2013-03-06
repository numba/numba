# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import fnmatch

from numba.typesystem.basetypes import *

def _typematch(pattern, typerepr):
    return fnmatch.fnmatch(typerepr, pattern)

def typematch(pattern, ty):
    """
    Match a type pattern to a type.

    >>> type = ListType(object_, 2)
    >>> typematch("list(*, 2)", type)
    True
    >>> typematch("list(*)", type)
    True
    >>> typematch("list(*)", type)
    True
    >>> typematch("tuple(*)", type)
    False
    >>> typematch("object_", type)
    True
    """
    return any(_typematch(pattern, cls.__repr__(ty))
                   for cls in type(ty).__mro__)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
