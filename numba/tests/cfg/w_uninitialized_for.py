# cython: warn.maybe_uninitialized=True
# mode: error
from numba import *

def simple_for(n):
    for i in range(n):
        a = 1
    return a

def simple_for_break(n):
    for i in range(n):
        a = 1
        break
    return a

def simple_for_pos(n):
    for i in range(n):
        a = 1
    else:
        a = 0
    return a

def simple_target(n):
    for i in range(n):
        pass
    return i

def simple_target_f(n):
    for i in range(n):
        i *= i
    return i

#def simple_for_from(n):
#    for i from 0 <= i <= n:
#        x = i
#    else:
#        return x

def for_continue(l):
    for i in range(l):
        if i > 0:
            continue
        x = i
    print x

def for_break(l):
    for i in range(l):
        if i > 0:
            break
        x = i
    print x

#def for_finally_continue(f):
#    for i in f:
#        try:
#            x = i()
#        finally:
#            print x
#            continue

def for_finally_break(f):
    for i in f:
        try:
            x = i()
        finally:
            print x
            break

def for_finally_outer(p, f):
    x = 1
    try:
        for i in f:
            print x
            x = i()
            if x > 0:
                continue
            if x < 0:
                break
    finally:
        del x

def jitfunc(func):
    jit(int_(int_))(func)

__doc__ = """
>>> jitfunc(simple_for)
Warning 8:11: local variable 'a' might be referenced before assignment
>>> jitfunc(simple_for_break)
Warning 14:11: local variable 'a' might be referenced before assignment
>>> jitfunc(simple_for_pos)
>>> jitfunc(simple_target)
Warning 26:11: local variable 'i' might be referenced before assignment
>>> jitfunc(simple_target_f)
Warning 31:11: local variable 'i' might be referenced before assignment
>>> jitfunc(for_continue)
Warning 44:10: local variable 'x' might be referenced before assignment
>>> jitfunc(for_break)
Warning 51:10: local variable 'x' might be referenced before assignment

Finally tests
>> jitfunc(for_finally_break)
Warning 58:19: local variable 'x' might be referenced before assignment
>> jitfunc(for_finally_outer)
Warning 66:19: local variable 'x' might be referenced before assignment
"""

if __name__ == "__main__":
    #    jitfunc(simple_for_break)
    #    jitfunc(simple_for_pos)
    import doctest
    doctest.testmod()