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

>>> eq("foo", "foo")
True
>>> eq("foo", "bar")
False
>>> ne("foo", "foo")
False
>>> ne("foo", "bar")
True
>>> lt("foo", "foo")
False
>>> lt("foo", "bar")
False
>>> lt("bar", "foo")
True

>>> interpolate("%s and %s", "ham", "eggs")
'ham and eggs'

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

@jit(bool_(c_string_type, c_string_type))
def eq(s1, s2):
    return s1 == s2

@jit(bool_(c_string_type, c_string_type))
def ne(s1, s2):
    return s1 != s2

@jit(bool_(c_string_type, c_string_type))
def lt(s1, s2):
    return s1 < s2

@jit(c_string_type(c_string_type, c_string_type))
def concat(s1, s2):
    return s1 + s2

@jit(c_string_type(c_string_type, c_string_type, c_string_type))
def interpolate(s, s1, s2):
    return s % (s1, s2)

def string_len(s):
    return len(s)

if __name__ == '__main__':
    import numba
    numba.testing.testmod()
