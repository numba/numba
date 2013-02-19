from numba import *
from numba import error

#@autojit
def func():
    if x:
        print "hello"
    else:
        print "world"

def compile_func1():
    try:
        jit(void())(func)
    except error.NumbaError, e:
        print "exception:", e

__doc__ = """
>>> compile_func1()
--------------------- Numba Encountered Errors or Warnings ---------------------
<BLANKLINE>
    if x:
-------^
Error 6:7: No global named 'x'
<BLANKLINE>
--------------------------------------------------------------------------------
exception: 6:7: No global named 'x'
"""

#@autojit
def func2():
    print 10[20]

def compile_func2():
    try:
        jit(void())(func2)
    except error.NumbaError, e:
        print "exception:", e

__doc__ += """>>> compile_func2()
--------------------- Numba Encountered Errors or Warnings ---------------------
<BLANKLINE>
    print 10[20]
----------^
Error 31:10: object of type int cannot be indexed
<BLANKLINE>
--------------------------------------------------------------------------------
exception: 31:10: object of type int cannot be indexed
"""

@autojit # this often messes up line numbers
def func_decorated():
    print 10[20]

def compile_func3():
    try:
        func_decorated()
    except error.NumbaError, e:
        print "exception:", e

__doc__ += """
>>> compile_func3()
--------------------- Numba Encountered Errors or Warnings ---------------------
<BLANKLINE>
    print 10[20]
----------^
Error 52:10: object of type int cannot be indexed
<BLANKLINE>
--------------------------------------------------------------------------------
exception: 52:10: object of type int cannot be indexed
"""

if __name__ == '__main__':
#    compile_func3()
    import doctest
    doctest.testmod()