from numba import *

@autojit
def error1():
    def inner():
        pass

@autojit
def error2():
    @autojit
    def inner():
        pass

@autojit
def closure1():
    a = 10
    @jit(argtypes=[int_])
    def inner(arg):
        print arg

    return inner

__doc__ = """
>>> error1()
Traceback (most recent call last):
    ...
NumbaError: 5:4: Closure must be decorated with 'jit' or 'autojit'
>>> error2()
Traceback (most recent call last):
    ...
NumbaError: 10:5: Dynamic closures not yet supported, use @jit
>>> closure1()
"""

if __name__ == '__main__':
#    closure1()
    import doctest
    doctest.testmod()