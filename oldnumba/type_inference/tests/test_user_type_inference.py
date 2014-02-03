from numba import *
from numba import register, register_callable, typeof, typeset

#----------------------------------------------------------------------------
# Type functions
#----------------------------------------------------------------------------

@register_callable(int32(double))
def func(arg):
    return int(arg)

@register_callable(typeset([int16(float_), int32(double), numeric(numeric)]))
def func_typeset_simple(arg):
    return int(arg)

@register_callable(numeric(numeric, numeric))
def func_typeset_binding(arg1, arg2):
    return int(arg1) + int(arg2)

#----------------------------------------------------------------------------
# Use of type functions
#----------------------------------------------------------------------------

@autojit
def use_user_type_function():
    return typeof(func(10.0))

@autojit
def use_typeset_function_simple():
    return typeof(func_typeset_simple(10.0))

@autojit
def use_typeset_function_binding(type1, type2):
    return typeof(func_typeset_binding(type1(10.0), type2(12.0)))

#----------------------------------------------------------------------------
# Test functions
#----------------------------------------------------------------------------

def test_register_callable():
    assert use_user_type_function() == int32
    assert use_typeset_function_simple() == int32

    assert use_typeset_function_binding(double, double) == double
    assert use_typeset_function_binding(float_, double) == double
    assert use_typeset_function_binding(int_, float_) == float_
    assert use_typeset_function_binding(int_, long_).itemsize == long_.itemsize

if __name__ == '__main__':
    test_register_callable()
