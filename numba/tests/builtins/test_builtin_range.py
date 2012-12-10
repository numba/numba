"""
>>> range_ret1()
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> range_ret2()
[1, 2, 3, 4]
>>> range_ret3()
[10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4]

>>> forward1()
0 1 2 3 4 5 6 7 8 9 done
>>> forward2()
1 2 3 4 done
>>> forward3()
5 8 11 14 done

>>> backward1()
10 7 4 done
>>> backward2()
done
>>> backward3()
-5 -8 -11 -14 done

>>> empty_assign()
14
>>> last_value()
Warning 92:10: local variable 'i' might be referenced before assignment
9
"""

from numba import *

@autojit
def range_ret1():
    return range(10)

@autojit
def range_ret2():
    return range(1, 5)

@autojit
def range_ret3():
    return range(10, -5, -1)

@autojit
def forward1():
    for i in range(10):
        print i,
    print "done"

@autojit
def forward2():
    for i in range(1, 5):
        print i,
    print "done"

@autojit
def forward3():
    for i in range(5, 15, 3):
        print i,
    print "done"

@autojit
def backward1():
    for i in range(10, 2, -3):
        print i,
    print "done"

@autojit
def backward2():
    for i in range(1, 5, -1):
        print i,
    print "done"

@autojit
def backward3():
    for i in range(-5, -15, -3):
        print i,
    print "done"

@autojit
def empty_assign():
    i = 14
    for i in range(10, 4):
        pass
    print i

@autojit
def last_value():
    for i in range(10):
        pass

    print i

if __name__ == '__main__':
    import doctest
    doctest.testmod()