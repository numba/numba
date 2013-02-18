"""
>>> compile_func1()
--------------------- Numba Encountered Errors or Warnings ---------------------
<BLANKLINE>
    if x:
-------^
Error 28:7: No global named 'x'
<BLANKLINE>
--------------------------------------------------------------------------------
exception: 28:7: No global named 'x'

>>> compile_func2()
--------------------- Numba Encountered Errors or Warnings ---------------------
<BLANKLINE>
    print 10[20]
----------^
Error 41:10: object of type int cannot be indexed
<BLANKLINE>
--------------------------------------------------------------------------------
exception: 41:10: object of type int cannot be indexed
"""

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

#@autojit
def func2():
    print 10[20]

def compile_func2():
    try:
        jit(void())(func2)
    except error.NumbaError, e:
        print "exception:", e


if __name__ == '__main__':
    compile_func1()
#    import doctest
#    doctest.testmod()