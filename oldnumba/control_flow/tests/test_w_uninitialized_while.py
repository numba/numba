from numba import *



def simple_while(n):
    while n > 0:
        n -= 1
        a = 0
    return a

def simple_while_break(n):
    while n > 0:
        n -= 1
        break
    else:
        a = 1
    return a

def simple_while_pos(n):
    while n > 0:
        n -= 1
        a = 0
    else:
        a = 1
    return a

#def while_finally_continue(p, f):
#    while p():
#        try:
#            x = f()
#        finally:
#            print x
#            continue
#
#def while_finally_break(p, f):
#    while p():
#        try:
#            x = f()
#        finally:
#            print x
#            break
#
#def while_finally_outer(p, f):
#    x = 1
#    try:
#        while p():
#            print x
#            x = f()
#            if x > 0:
#                continue
#            if x < 0:
#                break
#    finally:
#        del x

def jitfunc(func):
    jit(int_(int_), warnstyle='simple')(func)

__doc__ = """
>>> jitfunc(simple_while)
Warning 9:11: local variable 'a' might be referenced before assignment
>>> jitfunc(simple_while_break)
Warning 17:11: local variable 'a' might be referenced before assignment
>>> jitfunc(simple_while_pos)
"""

if __name__ == "__main__":
#    jitfunc(simple_while_break)
    import numba
    numba.testing.testmod()
