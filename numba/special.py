from __future__ import print_function, division, absolute_import

__all__ = [ 'typeof' ]

def typeof(val):
    """
    Get the type of a variable or value.

    Used outside of Numba code, infers the type for the object.
    """

    from .dispatcher import typeof_pyval
    return typeof_pyval(val)

