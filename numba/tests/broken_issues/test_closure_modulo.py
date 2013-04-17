from numba import autojit
import numpy as np
print np.zeros(10).dtype


@autojit
def closure_modulo(a, b):
    c = np.zeros(10)
    @jit('f8[:]()')
    def foo():
        c[0] = a % b
        return c
    return foo()

print closure_modulo(100, 48)
