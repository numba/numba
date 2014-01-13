from numba import *

@jit(void())
def func1():
    "I am a docstring!"

@autojit
def func2():
    "I am a docstring!"


if __name__ == '__main__':
    assert func1.__name__ == func1.py_func.__name__
    assert func1.__doc__ == "I am a docstring!"
    assert func1.__module__ == func1.py_func.__module__

    assert func2.__name__ == func2.py_func.__name__
    assert func2.__doc__ == "I am a docstring!", func2.__doc__
    # This does not yet work for some reason (maybe overridden by PyType_Ready()?)
    # assert func2.__module__ == func2.py_func.__module__
