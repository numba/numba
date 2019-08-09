""" Test cases for inlining IR from another module """
from numba import njit

_GLOBAL1 = 100


@njit(inline='always')
def bar():
    return _GLOBAL1 + 10


def baz_factory(a):
    b = 17 + a
    @njit(inline='always')
    def baz():
        return _GLOBAL1 + a - b
    return baz
