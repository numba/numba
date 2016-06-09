"""
Usecases of recursive functions.

Some functions are compiled at import time, hence a separate module.
"""

from numba import jit


@jit("i8(i8)", nopython=True)
def fib1(n):
    if n < 2:
        return n
    # Note the second call uses a named argument
    return fib1(n - 1) + fib1(n=n - 2)


def make_fib2():
    @jit("i8(i8)", nopython=True)
    def fib2(n):
        if n < 2:
            return n
        return fib2(n - 1) + fib2(n=n - 2)

    return fib2

fib2 = make_fib2()


# Implicit signature (unsupported)
@jit(nopython=True)
def fib3(n):
    if n < 2:
        return n
    return fib3(n - 1) + fib3(n - 2)


# Mutual recursion (unsupported)
@jit(nopython=True)
def outer_fac(n):
    if n < 1:
        return 1
    return n * inner_fac(n - 1)

@jit(nopython=True)
def inner_fac(n):
    if n < 1:
        return 1
    return n * outer_fac(n - 1)

