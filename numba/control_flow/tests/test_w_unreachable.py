from numba import *

jitv = jit(void(), warnstyle='simple') #, nopython=True)




def simple_return():
    """
    >>> result = jitv(simple_return)
    Warning 14:4: Unreachable code
    """
    return
    print('Where am I?')

def simple_loops():
    """
    >>> result = jitv(simple_loops)
    Warning 28:8: Unreachable code
    Warning 32:8: Unreachable code
    Warning 36:8: Unreachable code
    Warning 41:12: Unreachable code
    Warning 46:8: Unreachable code
    Warning 50:4: Unreachable code
    """
    for i in range(10):
        continue
        print('Never be here')

    while True:
        break
        print('Never be here')

    while True:
        break
        print('Never be here')

    for i in range(10):
        for j in range(10):
            return
            print("unreachable")
        else:
            print("unreachable")
        print("unreachable")
        return
        print("unreachable")

    print("unreachable")
    return
    print("unreachable")

def conditional(a, b):
    if a:
        return 1
    elif b:
        return 2
    else:
        return 37
    print('oops')

if __name__ == "__main__":
#    jitv(simple_loops)
#    jitv(simple_return)
    import numba
    numba.testmod()
