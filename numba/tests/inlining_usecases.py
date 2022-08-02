""" Test cases for inlining IR from another module """
from numba import njit
from numba.core.extending import overload

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


def baz():
    return _GLOBAL1 + 10


@overload(baz, inline='always')
def baz_ol():
    def impl():
        return _GLOBAL1 + 10
    return impl


def bop_factory(a):
    b = 17 + a

    def bop():
        return _GLOBAL1 + a - b

    @overload(bop, inline='always')
    def baz():
        def impl():
            return _GLOBAL1 + a - b
        return impl

    return bop
