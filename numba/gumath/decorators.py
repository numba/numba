import functools

from .ndtypes import Function
from .gufunc import GuFunc


def jit_xnd(fn_or_types=None, *, types=None, **kwargs):
    """
    Can either directly decorate a function, like this:

    >>> @jit_xnd
        def add(a, b):
            return a + b
    >>> add(xnd(1), xnd(2))
    xnd(3)

    In which case it will compile a function based on the type of the arguments
    you pass in, assuming the function operates only on the innermost dimensions
    (is zero dimensional).

    Or you can register the types directly when you decorate the function:

    >>> @jit_xnd('... * int64, ... * int64 -> ... * int64')
    def add(a, b):
        return a + b
    >>> add(xnd(1), xnd(2))
    xnd(3)

    You can also pass in kwargs in either mode, which are passed down to `jit`.
    """

    # we were passed in types, instead of a function
    if not callable(fn_or_types):
        # if we passed in types as the first argument, we shouldn't have
        # passed in the types kwarg
        assert types is None
        return functools.partial(jit_xnd, types=fn_or_types, **kwargs)
    if not fn_or_types:
        print(fn_or_types, types, kwargs)
    fn = fn_or_types
    g = GuFunc(fn, **kwargs)

    # allow single type instead of list
    if isinstance(types, str) or isinstance(types, Function):
        types = [types]
    
    # if we didn't get any types, then assume the user
    # wanted to infer them from the input types at call time.
    if types is None:
        return g

    # parse string types
    for t in types:
        if isinstance(t, str):
            t = Function.from_ndt(t)
        g.compile(t)
    return g.func
