"""
Defines the typeset class and a number of builtin type sets.
"""

from numba.typesystem import basetypes
from numba.minivect import minitypes

__all__ = [ 'typeset', 'numeric', 'integral', 'floating', 'complex' ]

class typeset(object):
    """
    Holds a set of types that can be used to specify signatures for
    type inference.
    """

    def __init__(self, types, name=None):
        self.types = types
        self.name = name

    def __repr__(self):
        return "typeset(%s)" % self.types

    def __hash__(self):
        return hash(id(self))


numeric = typeset(minitypes.numeric)
integral = typeset(minitypes.integral)
floating = typeset(minitypes.floating)
complex = typeset(minitypes.complextypes)
