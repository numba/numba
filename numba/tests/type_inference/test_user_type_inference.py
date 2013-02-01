from numba import *
from numba import register, register_callable

@register_callable(int32(double))
def func(arg):
    return int(arg)



def test_register_callable():
    assert test_user_type_function()