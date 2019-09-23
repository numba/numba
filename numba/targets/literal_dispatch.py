from numba.extending import overload
from numba import types
from numba.special import literally
from numba.errors import TypingError


@overload(literally)
def _ov_literally(obj):
    if isinstance(obj, types.Literal):
        lit = obj.literal_value
        return lambda obj: lit
    else:
        m = "Invalid use of non-Literal type in literally({})".format(obj)
        raise TypingError(m)
