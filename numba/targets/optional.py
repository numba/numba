
from __future__ import print_function, absolute_import, division
from numba import types, cgutils


def make_optional(valtype):
    """
    Return the Structure representation of the given *array_type*
    (an instance of types.Array).
    """
    class OptionalStruct(cgutils.Structure):
        _fields = [('data', valtype),
                   ('valid', types.boolean)]

    return OptionalStruct
