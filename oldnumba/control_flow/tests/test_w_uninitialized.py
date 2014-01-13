from numba import *

jitv = jit(void(), warnstyle='simple')
jitvi = jit(void(int_), warnstyle='simple')
jitvii = jit(void(int_, int_), warnstyle='simple')
jitii = jit(int_(int_), warnstyle='simple')
jitiii = jit(int_(int_, int_), warnstyle='simple')


def simple():
    """
    >>> jitv(simple)
    Traceback (most recent call last):
        ...
    NumbaError: 17:10: local variable 'a' referenced before assignment
    """
    print(a)
    a = 0

def simple2(arg):
    """
    >>> result = jitii(simple2)
    Warning 27:11: local variable 'a' might be referenced before assignment
    """
    if arg > 0:
        a = 1
    return a

def simple_pos(arg):
    """
    >>> result = jitii(simple_pos)
    """
    if arg > 0:
        a = 1
    else:
        a = 0
    return a

def ifelif(c1, c2):
    """
    >>> result = jitiii(ifelif)
    Warning 51:11: local variable 'a' might be referenced before assignment
    """
    if c1 == 1:
        if c2:
            a = 1
        else:
            a = 2
    elif c1 == 2:
        a = 3
    return a

def nowimpossible(a):
    """
    >>> result = jitvi(nowimpossible)
    Warning 61:14: local variable 'b' might be referenced before assignment
    """
    if a:
        b = 1
    if a:
        print(b)

def fromclosure():
    """
    >> result = jitv(fromclosure)
    """
    def bar():
        print(a)
    a = 1
    return bar

# Should work ok in both py2 and py3
def list_comp(a):
    return [i for i in a]

def set_comp(a):
    return set(i for i in a)

#def dict_comp(a):
#    return {i: j for i, j in a}

# args and kwargs
def generic_args_call(*args, **kwargs):
    return args, kwargs

def cascaded(x):
    print((a, b))
    a = b = x

def from_import():
    print(bar)
    from foo import bar

def regular_import():
    print(foo)
    import foo

def raise_stat():
    try:
        raise exc(msg)
    except:
        pass
    exc = ValueError
    msg = 'dummy'

def defnode_decorator():
    @decorator
    def foo():
        pass
    def decorator():
        pass

def defnode_default():
    def foo(arg=default()):
        pass
    def default():
        pass

def class_bases():
    class foo(bar):
        pass
    class bar(object):
        pass

def class_decorators():
    @decorator
    class foo(object):
        pass
    def decorator(cls):
        return cls

def uninitialized_augmented_assignment():
    """
    >>> func = jitv(uninitialized_augmented_assignment)
    Traceback (most recent call last):
        ...
    NumbaError: 139:4: local variable 'x' referenced before assignment
    """
    x += 1


def uninitialized_augmented_assignment_loop():
    """
    >>> func = jitv(uninitialized_augmented_assignment_loop)
    Warning 148:8: local variable 'x' might be referenced before assignment
    """
    for i in range(10):
        x += 1

    x = 0

if __name__ == "__main__":
    import numba
    numba.testing.testmod()
