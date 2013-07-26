"""
>>> temp_string_var()
hellohello0
>>> temp_string()
hellohello0
>>> temp_string2()
hellohello0
>>> temp_string3()
hellohello0
hellohello1
hellohello2

>>> autojit(string_len)("hello")
5
>>> autojit(nopython=True)(string_len)("hello")
5
"""

import sys

from numba import *

def get_string(i=0):
    s = "hello"
    return s * 2 + str(i)

@autojit(backend='ast', locals=dict(s=c_string_type))
def temp_string_var():
    s = get_string()
    print(s)

@autojit(backend='ast', locals=dict(s=c_string_type))
def temp_string():
    s = c_string_type(get_string())
    print(s)

@autojit(backend='ast')
def temp_string2():
    print((c_string_type(get_string())))

@autojit(backend='ast', locals=dict(s=c_string_type))
def temp_string3():
    for i in range(3):
        s = c_string_type(get_string(i))
        print(s)

@autojit(backend='ast')
def test():
    return object()

@jit(void())
def string_constant():
    print("hello world")

def string_len(s):
    return len(s)

if __name__ == '__main__':
    string_constant()

    import numba
    numba.testing.testmod()
