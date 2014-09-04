from __future__ import print_function, division, absolute_import

__all__ = [ 'typeof' ]

def typeof(val):
    """
    Get the type of a variable or value.

    Used outside of Numba code, infers the type for the object.
    """
    from .targets.registry import CPUTarget
    return CPUTarget.typing_context.resolve_data_type(val)

