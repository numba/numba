import functools
import collections.abc

from .ndtypes import Function
from .gufunc import GuFunc

@functools.singledispatch
def vectorize(fn):
    """
    Can either directly decorate a function, like this:

    >>> @vectorize
        def add(a, b):
            return a + b
    >>> add(xnd(1), xnd(2))
    xnd(3)

    In which case it will compile a function based on the type of the arguments
    you pass in, assuming the function operates only on the innermost dimensions
    (is zero dimensional). Here, `add` will be a `GuFunc` class.

    Or you can register the types directly when you decorate the function:

    >>> @vectorize('... * int64, ... * int64 -> ... * int64')
    def add(a, b):
        return a + b
    >>> add(xnd(1), xnd(2))
    xnd(3)

    Then, `add` will be a `_gumath.gufunc`.

    You can also past in a list of signatures, instead of just one. 
    """
    return _vecorize_fn(None, fn)

@vectorize.register(collections.abc.Iterable)
def _(types):
    return functools.partial(_vecorize_fn, types)

@vectorize.register(str)
def _(type):
    return vectorize([type])


def _vecorize_fn(types, fn):
    g = GuFunc(fn)

    # if we didn't get any types, then assume the user
    # wanted to infer them from the input types at call time.
    if types is None:
        return g

    for t in types:
        g.compile(Function.from_ndt(t))
    # Return gufunc with kernels already compiled
    return g.func
