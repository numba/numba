from numba import *
from numba import error

#@autojit
def func():
    if x:
        print("hello")
    else:
        print("world")

def compile_func1():
    try:
        jit(void())(func)
    except error.NumbaError as e:
        print("exception: %s" % e)

__doc__ = """
>>> compile_func1()
exception: (see below)
--------------------- Numba Encountered Errors or Warnings ---------------------
    if x:
-------^
Error ...: No global named 'x'
--------------------------------------------------------------------------------
"""

#@autojit
def func2():
    print(10[20])

def compile_func2():
    try:
        jit(void())(func2)
    except error.NumbaError as e:
        print("exception: %s" % e)

__doc__ += """>>> compile_func2()
exception: (see below)
--------------------- Numba Encountered Errors or Warnings ---------------------
    print(10[20])
----------^
Error ...: object of type int cannot be indexed
--------------------------------------------------------------------------------
"""

@autojit # this often messes up line numbers
def func_decorated():
    print(10[20])

def compile_func3():
    try:
        func_decorated()
    except error.NumbaError as e:
        print("exception: %s" % e)

__doc__ += """
>>> compile_func3()
exception: (see below)
--------------------- Numba Encountered Errors or Warnings ---------------------
    print(10[20])
----------^
Error ...: object of type int cannot be indexed
--------------------------------------------------------------------------------
"""

def warn_and_error(a, b):
    print(a)
    1[2]

__doc__ += """
>>> autojit(warn=False)(warn_and_error)(1, 2)
Traceback (most recent call last):
    ...
NumbaError: (see below)
--------------------- Numba Encountered Errors or Warnings ---------------------
    1[2]
----^
Error 68:4: object of type int cannot be indexed
<BLANKLINE>
--------------------------------------------------------------------------------

>>> autojit(warnstyle='simple')(warn_and_error)(1, 2)
Traceback (most recent call last):
    ...
NumbaError: (see below)
Error ...: object of type int cannot be indexed
Warning ...: Unused argument 'b'

>>> autojit(func_decorated.py_func, warnstyle='simple')()
Traceback (most recent call last):
    ...
NumbaError: ...: object of type int cannot be indexed
"""

if __name__ == '__main__':
    import numba
    numba.testing.testmod()
