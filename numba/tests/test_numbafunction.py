from numba import *

@jit(void())
def func1():
    "I am a docstring!"

@autojit
def func2():
    "I am a docstring!"

if __name__ == '__main__':
    assert func1.__doc__ == "I am a docstring!"
    # assert func2.__doc__ == "I am a docstring!", func2.__doc__