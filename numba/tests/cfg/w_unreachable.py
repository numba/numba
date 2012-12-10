from numba import *

jitv = jit(void()) #, nopython=True)




def simple_return():
    """
    >>> result = jitv(simple_return)
    Warning, unreachable code at 14:4
    """
    return
    print 'Where am I?'

def simple_loops():
    """
    >>> result = jitv(simple_loops)
    Warning, unreachable code at 28:8
    Warning, unreachable code at 32:8
    Warning, unreachable code at 36:8
    Warning, unreachable code at 41:12
    Warning, unreachable code at 46:8
    Warning, unreachable code at 50:4
    """
    for i in range(10):
        continue
        print 'Never be here'

    while True:
        break
        print 'Never be here'

    while 1:
        break
        print 'Never be here'

    for i in range(10):
        for j in range(10):
            return
            print "unreachable"
        else:
            print "unreachable"
        print "unreachable"
        return
        print "unreachable"

    print "unreachable"
    return
    print "unreachable"

def conditional(a, b):
    if a:
        return 1
    elif b:
        return 2
    else:
        return 37
    print 'oops'

if __name__ == "__main__":
#    jitv(simple_loops)
#    jitv(simple_return)
    import doctest
    doctest.testmod()
