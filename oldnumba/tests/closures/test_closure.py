import numba
from numba import *
from numba.error import NumbaError

autojit = autojit(warn=False, warnstyle='simple')

@autojit
def error1():
    def inner():
        pass

@autojit
def error2():
    @autojit
    def inner():
        pass

@autojit
def error3():
    inner(10, 20, 30)

    @jit(restype=void, argtypes=[int_, int_, int_])
    def inner(a, b, c):
        print(str(a) + ' ' + str(b) + ' ' + str(c))

@autojit
def error4():
    @jit(restype=void, argtypes=[int_, int_, int_])
    def inner(a, b, c):
        print(str(a) + ' ' + str(b) + ' ' + str(c))

    inner(10, a=20, b=30, c=40)

@autojit
def error5():
    @jit(restype=void, argtypes=[int_, int_, int_])
    def inner(a, b, c):
        print(str(a) + ' ' + str(b) + ' ' + str(c))

    inner(10, a=20, b=30)

@autojit
def closure1():
    a = 10
    @jit(restype=void, argtypes=[int_])
    def inner(arg):
        print(str(arg))

    return inner

@autojit
def closure2():
    a = 10
    @jit(restype=void, argtypes=[int_])
    def inner(arg):
        print(str(arg))

    inner(arg=a)

@autojit
def closure3():
    a = 10

    @jit('void()')
    def inner():
        print(a)

    a = 12
    inner()

@autojit
def closure4():
    a = 10

    @jit('void()')
    def inner():
        print(a)

    a = 12
    return inner

@autojit
def nested_closure():
    a = 20
    b = 21

    @jit('object_()')
    def c1():
        b = 10
        @jit('void()')
        def c2():
            print(str(a) + ' ' + str(b))

        return c2

    @jit('void()')
    def c3():
        print(b)

    return c1, c3

__doc__ = """
>>> error1()
Traceback (most recent call last):
    ...
NumbaError: ...: Closure must be decorated with 'jit' or 'autojit'
>>> error2()
Traceback (most recent call last):
    ...
NumbaError: ...: Dynamic closures not yet supported, use @jit
>>> error3()
Traceback (most recent call last):
    ...
NumbaError: ...: local variable 'inner' referenced before assignment
>>> error4()
Traceback (most recent call last):
    ...
NumbaError: ...: Expected 3 arguments, got 4
>>> error5()
Traceback (most recent call last):
    ...
NumbaError: ...: Got multiple values for positional argument 'a'

Test closures

>>> closure1().__name__
'inner'
>>> closure1()()
Traceback (most recent call last):
    ...
TypeError: function takes exactly 1 argument (0 given)
>>> closure1()(object())
Traceback (most recent call last):
    ...
TypeError: an integer is required
>>> closure1()(10.0)
10
>>> closure2()
10
>>> closure3()
12
>>> func = closure4()
>>> print(func.__name__)
inner
>>> field, = func.__closure__._numba_attrs._fields_
>>> import ctypes
>>> print((field[0], field[1] == ctypes.c_int))
('a', True)
>>> print(func.__closure__._numba_attrs.a)
12
>>> func()
12

>>> c1, c3 = nested_closure()
>>> c1.__name__
'c1'
>>> c3.__name__
'c3'
>>> c1().__name__
'c2'
>>> c1()()
20 10
>>> c3()
21
"""

@autojit
def closure_arg(a):
    @jit('object_(object_)')
    def closure1(b):
        print(str(a) + ' ' + str(b))
        x = 10 + int_(b)
        @jit('object_(object_)')
        def closure2(c):
            print(str(a) + ' ' + str(b) + ' ' + str(c) + ' ' + str(x))
            y = double(x) + double(c)
            @jit('void(object_)')
            def closure3(d):
                print(str(a) + ' ' + str(b) + ' ' + str(c) + ' ' + str(d) + ' ' + str(x) + ' ' + str(y))
            return closure3
        return closure2
    return closure1

__doc__ += \
"""
>>> closure1 = closure_arg(1)
>>> closure1.__name__
'closure1'

>>> closure2_1 = closure1(2)
1 2
>>> closure2_1.__name__
'closure2'
>>> closure2_2 = closure1(3)
1 3
>>> closure2_2.__name__
'closure2'

>>> closure3_1 = closure2_1(4)
1 2 4 12
>>> closure3_1.__name__
'closure3'
>>> closure3_2 = closure2_2(5)
1 3 5 13
>>> closure3_2.__name__
'closure3'

>>> closure3_1(6)
1 2 4 6 12 16.0
>>> closure3_2(7)
1 3 5 7 13 18.0
"""


@autojit
def closure_arg_simple(a):
    @jit('object_(object_)')
    def inner(b):
        print(str(a) + ' ' + str(b))
        @jit('void(object_)')
        def inner_inner(c):
            print(str(a) + ' ' + str(b) + ' ' + str(c))
        return inner_inner
    return inner

__doc__ += """
>>> closure_arg_simple(10)(20)(30)
10 20
10 20 30
"""

@autojit
def closure_skip_level(a):
    @jit('object_()')
    def inner():
        @jit('void()')
        def inner_inner():
            print(str(a))
        return inner_inner
    return inner

__doc__ += """
>>> closure_skip_level(10)()()
10
"""

@autojit
def objects(s):
    @jit('object_()')
    def inner():
        return s.upper()
    return inner

__doc__ += """
>>> objects("hello")()
'HELLO'
"""

@autojit
def wrong_signature(s):
    @jit('object_(object_)')
    def inner():
        return s.upper()
    return inner

__doc__ += """
>>> try_(wrong_signature, "foo")
NumbaError: ...: Expected 1 arguments type(s), got 0
"""

@autojit
def wrong_restype():
    @jit('object_()')
    def inner():
        pass
    return inner

__doc__ += """
>>> try_(wrong_restype)
NumbaError: ...: Function with non-void return does not return a value
"""

#
### Test signatures like @double(object_)
#
@autojit
def signature_dec():
    @object_()
    def inner():
        return "hello"
    return inner

__doc__ += """
>>> signature_dec()()
'hello'
"""

@autojit
def wrong_signature2(s):
    @object_(object_)
    def inner():
        return s.upper()
    return inner

__doc__ += """
>>> try_(wrong_signature2, "foo")
NumbaError: ...: Expected 1 arguments type(s), got 0
"""

@autojit
def get_closure(arg):
    @void()
    def closure():
        print(arg)

    closure()
    return closure

@autojit
def test_call_closure():
    closure = get_closure(10.0)
    closure() # TODO: This still goes through the object layer, amend

__doc__ += """
>>> test_call_closure()
10.0
10.0
"""

@autojit
def test_call_closure_from_closure():
    closure = get_closure(10.0)
    @void()
    def inner():
        closure()
    return inner

__doc__ += """
>>> test_call_closure_from_closure()()
10.0
10.0
"""

@autojit
def test_closure_loop():
    """
    >>> test_closure_loop()
    0 3
    1 3
    2 3
    <BLANKLINE>
    0 3
    1 3
    2 3
    """
    cellvar = 3

    @jit(void())
    def inner():
        for i in range(cellvar):
            print(str(i) + ' ' + str(cellvar))

        print('')

        for i in range(cellvar):
            for j in range(cellvar):
                if i == j:
                    print(str(i) + ' ' + str(cellvar))

    inner()

@numba.autojit(locals=dict(var=int_), warn=False)
def test_closure_outer_locals():
    """
    >>> test_closure_outer_locals()
    """
    var = 10
    @jit(void())
    def inner():
        var = "hello"

    inner()

#__doc__ = rewrite_doc(__doc__)

def try_(func, *args):
    try:
        func(*args)
    except NumbaError as e:
        print("%s%s: %s" % ('numba.error.' if numba.PY3 else '',
                            type(e).__name__, e))

if __name__ == '__main__':
#    closure1 = closure_arg(1)
    # print closure1.__name__
    # closure1(10)
    # test_call_closure()
    # closure4()
    # signature_dec()()
    # test_closure_outer_locals()
    # test_closure_loop()
    # test_closure_outer_locals()
#    test_call_closure_from_closure()()
    # wrong_restype()
    import numba
    numba.testing.testmod()
