from numba import *
from numba import register, register_callable, typeof

@register_callable(int32(double))
def func(arg):
    return int(arg)

@autojit
def use_user_type_function():
    return typeof(func(10.0))

def test_register_callable():
    assert use_user_type_function() == int32

if __name__ == '__main__':
    test_register_callable()