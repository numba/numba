from numba.extending import overload
from numba import types
from numba.special import literally


@overload(literally)
def _ov_literally(obj):
    if isinstance(obj, types.Literal):
        lit = obj.literal_value
        return lambda obj: lit
