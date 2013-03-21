from numba import *

a = 10
b = 11
c = 12

def jitter():
    a = 20
    b = 21
    c = 22

    @jit(object_())
    def func():
        return a, c

    return func

func = jitter()
assert func() == (20, 22)
