"""
Special compiler-recognized numba functions and attributes.
"""

__all__ = ['NULL', 'typeof']

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
