from numba import *

@jit(void())
def func1():
    "I am a docstring!"

@autojit
def func2():
    "I am a docstring!"

def module_from_name(func):
    return ".".join(func.__name__.split(".")[:-1])

def name(func):
    return func.__name__.split(".")[-1]

if __name__ == '__main__':
    assert name(func1) == "func1"
    assert func1.__doc__ == "I am a docstring!"
    assert func1.__module__ == module_from_name(func1)

    assert name(func2) == "func2"
    assert func2.__doc__ == "I am a docstring!"
    # This does not yet work for some reason (maybe overridden by PyType_Ready()?)
    # assert func2.__module__ == module_from_name(func2), func2.__module__
