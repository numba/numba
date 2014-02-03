# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import fnmatch

def _typematch(pattern, typerepr):
    return fnmatch.fnmatch(typerepr, pattern)

def typematch(pattern, ty):
    """
    Match a type pattern to a type.

    >>> type = list_(object_, 2)
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
    return (_typematch(pattern, repr(ty)) or
            any(_typematch(pattern, flag) for flag in ty.flags))

if __name__ == '__main__':
    import doctest
    doctest.testmod()
