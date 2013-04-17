from numba import autojit

@autojit
def closure_modulo(a, b):
    @jit('int32()')
    def foo():
        return a % b
    return foo()

print closure_modulo(100, 48)
