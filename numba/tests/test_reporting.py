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
--------------------- Numba Encountered Errors or Warnings ---------------------
    if x:
-------^
Error 6:7: No global named 'x'
--------------------------------------------------------------------------------
exception: 6:7: No global named 'x'
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
--------------------- Numba Encountered Errors or Warnings ---------------------
    print(10[20])
----------^
Error 29:10: object of type int cannot be indexed
--------------------------------------------------------------------------------
exception: 29:10: object of type int cannot be indexed
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
--------------------- Numba Encountered Errors or Warnings ---------------------
    print(10[20])
----------^
Error 48:10: object of type int cannot be indexed
--------------------------------------------------------------------------------
exception: 48:10: object of type int cannot be indexed
"""

if __name__ == '__main__':
    import numba
    numba.testing.testmod()
