import sys

from numba.extending import overload
from numba import types
from numba.special import literally, literal_unroll
from numba.errors import TypingError


@overload(literally)
def _ov_literally(obj):
    if isinstance(obj, types.Literal):
        lit = obj.literal_value
        return lambda obj: lit
    else:
        m = "Invalid use of non-Literal type in literally({})".format(obj)
        raise TypingError(m)


@overload(literal_unroll)
def literal_unroll_impl(container):
    from numba.errors import UnsupportedError
    if sys.version_info[:2] < (3, 6):
        raise UnsupportedError("literal_unroll is only support in Python > 3.5")

    def impl(container):
        return container
    return impl
