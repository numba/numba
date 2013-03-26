# -*- coding: utf-8 -*-
"""
Special compiler-recognized numba functions and attributes.
"""
from __future__ import print_function, division, absolute_import

__all__ = ['NULL', 'typeof', 'python', 'nopython']

class NumbaDotNULL(object):
    "NULL pointer"

NULL = NumbaDotNULL()

def typeof(variable):
    """
    Get the type of a variable.

    Used outside of Numba code, infers the type for the object.
    """
    from numba.environment import NumbaEnvironment
    context = NumbaEnvironment.get_environment().context
    return context.typemapper.from_python(variable)

class NoopContext(object):

    def __init__(self, name):
        self.name = name

    def __enter__(self, *args):
        return None

    def __exit__(self, *args):
        return None

    def __repr__(self):
        return self.name

python = NoopContext("python")
nopython = NoopContext("nopython")